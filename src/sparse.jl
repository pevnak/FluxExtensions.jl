import Flux.Tracker.back
import Base.*

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

a::Flux.Tracker.TrackedMatrix * b::SparseMatrixCSC = Flux.Tracker.track(mul, a, b)
Flux.Tracker.@grad function mul(a::AbstractMatrix, b::SparseMatrixCSC)
  return mul(Flux.data(a),b) , Δ -> (multrans(Δ, b),nothing)
end

# using CuArrays
# using CUSPARSE
# CuArrays.allowscalar(false)
# back(::typeof(*), Δ, a::CuMatrix, b::CudaSparseMatrixCSC) = Flux.Tracker.@back(a, transpose(A_mul_Bt(b,Δ)))
# back(::typeof(mul), Δ, a::CuMatrix, b::CudaSparseMatrixCSC) = Flux.Tracker.@back(a, transpose(A_mul_Bt(b,Δ)))
# mul(A::CuMatrix{T},B::CudaSparseMatrixCSC{T}) where T = transpose(At_mul_Bt(A,B))
# a::Flux.Tracker.TrackedMatrix * b::CudaSparseMatrixCSC = Flux.Tracker.TrackedArray(Flux.Tracker.Call(mul, a, b))