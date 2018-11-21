import Flux.Tracker.back
import Base.*
using SparseArrays

function multrans(A::Matrix, B::SparseMatrixCSC)
  mA, nA = size(A)
  mB, nB = size(B)
  @boundscheck nA == nB || throw(DimensionMismatch())
  C = zeros(promote_type(eltype(A),eltype(B)), mA, mB)
  @inbounds for jB in 1:nB
      for kB in B.colptr[jB]:(B.colptr[jB+1] - 1)
          iB = B.rowval[kB]
          xB = B.nzval[kB]
          for iA in 1:mA
              C[iA, iB] += A[iA, jB] * xB
          end
      end
  end
  return C
end

function mul(A::Matrix, B::SparseMatrixCSC)
  mA, nA = size(A)
  mB, nB = size(B)
  @boundscheck nA == mB || throw(DimensionMismatch())
  C = zeros(promote_type(eltype(A),eltype(B)), mA, nB)
  @inbounds for jB in 1:nB
      for kB in B.colptr[jB]:(B.colptr[jB+1] - 1)
          iB = B.rowval[kB]
          xB = B.nzval[kB]
          for iA in 1:mA
              C[iA, jB] += A[iA, iB] * xB
          end
      end
  end
  return C
end

"""
  mul(A::Matrix, B::SparseMatrixCSC, mask::Vector{Int})
  mul(A::Matrix, B::SparseMatrixCSC, mask::BitArray)

  multiply transposed matrix `A` with `B` while assuming that only rows of `B` with indexes in `mask` are non-zeros
"""
function multrans(A::Matrix, B::SparseMatrixCSC, mask::BitArray)
  mA, nA = size(A)
  mB, nB = size(B)
  @boundscheck nA == nB || throw(DimensionMismatch())
  C = zeros(promote_type(eltype(A),eltype(B)), mA, mB)
  @inbounds for jB in 1:nB
      for kB in B.colptr[jB]:(B.colptr[jB+1] - 1)
          iB = B.rowval[kB]
          xB = B.nzval[kB]
          !mask[iB] && continue
          for iA in 1:mA
              C[iA, iB] += A[iA, jB] * xB
          end
      end
  end
  return C
end
multrans(A::Matrix, B::SparseMatrixCSC, mask::Vector{Int}) = (m = falses(size(A,2)); m[mask] .= true; multrans(A, B, m))

"""
  mul(A::Matrix, B::SparseMatrixCSC, mask::Vector{Int})
  mul(A::Matrix, B::SparseMatrixCSC, mask::BitArray)

  multiply the matrix `A` with `B` while assuming that only rows of `B`  with indexes in `mask` are non-zeros
"""
function mul(A::Matrix, B::SparseMatrixCSC, mask::BitArray)
  mA, nA = size(A)
  mB, nB = size(B)
  @boundscheck nA == mB || throw(DimensionMismatch())
  C = zeros(promote_type(eltype(A),eltype(B)), mA, nB)
  @inbounds for jB in 1:nB
      for kB in B.colptr[jB]:(B.colptr[jB+1] - 1)
          iB = B.rowval[kB]
          xB = B.nzval[kB]
          !mask[iB] && continue
          for iA in 1:mA
              C[iA, jB] += A[iA, iB] * xB
          end
      end
  end
  return C
end
mul(A::Matrix, B::SparseMatrixCSC, mask::Vector{Int}) = (m = falses(size(A,2)); m[mask] .= true; mul(A, B, m))


a::Flux.Tracker.TrackedMatrix * b::SparseMatrixCSC = Flux.Tracker.track(mul, a, b)
Flux.Tracker.@grad function mul(a::AbstractMatrix, b::SparseMatrixCSC)
  return mul(Flux.data(a),b) , Δ -> (multrans(Δ, b),nothing)
end

mul(a::Flux.Tracker.TrackedMatrix, b::SparseMatrixCSC, mask) = Flux.Tracker.track(mul, a, b, mask)
Flux.Tracker.@grad function mul(a::AbstractMatrix, b::SparseMatrixCSC, mask)
  return mul(Flux.data(a), b, mask) , Δ -> (multrans(Δ, b),nothing, nothing)
end

function SparseArrays.findnz(S::SparseMatrixCSC{Tv,Ti}, coloffset) where {Tv,Ti}
    numnz = nnz(S)
    I = Vector{Ti}(undef, numnz)
    J = Vector{Ti}(undef, numnz)
    V = Vector{Tv}(undef, numnz)

    count = 1
    @inbounds for col = 1 : S.n, k = S.colptr[col] : (S.colptr[col+1]-1)
        I[count] = S.rowval[k]
        J[count] = col + coloffset
        V[count] = S.nzval[k]
        count += 1
    end

    return (I, J, V)
end

function Base.reduce(::typeof(hcat), A::AbstractVector{T}) where {T<: SparseMatrixCSC{Tv, Ti} where {Tv, Ti}}
  length(A) <= 1 && return(A)
  Tv, Ti = eltype(A[1].nzval), eltype(A[1].colptr)
  coloffset = 0
  Is, Js, Vs = Vector{Vector{Ti}}(undef, length(A)), Vector{Vector{Ti}}(undef, length(A)), Vector{Vector{Tv}}(undef, length(A))
  for (i, a) in enumerate(A)
    (I, J, V) = findnz(a, coloffset)
    Is[i] =  I
    Js[i] =  J
    Vs[i] =  V
    coloffset += a.n
  end
  sparse(reduce(vcat, Is), reduce(vcat, Js), reduce(vcat, Vs),  A[1].m, coloffset)
end


# using CuArrays
# using CUSPARSE
# CuArrays.allowscalar(false)
# back(::typeof(*), Δ, a::CuMatrix, b::CudaSparseMatrixCSC) = Flux.Tracker.@back(a, transpose(A_mul_Bt(b,Δ)))
# back(::typeof(mul), Δ, a::CuMatrix, b::CudaSparseMatrixCSC) = Flux.Tracker.@back(a, transpose(A_mul_Bt(b,Δ)))
# mul(A::CuMatrix{T},B::CudaSparseMatrixCSC{T}) where T = transpose(At_mul_Bt(A,B))
# a::Flux.Tracker.TrackedMatrix * b::CudaSparseMatrixCSC = Flux.Tracker.TrackedArray(Flux.Tracker.Call(mul, a, b))