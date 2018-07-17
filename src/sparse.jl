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

a::Flux.Tracker.TrackedMatrix * b::SparseMatrixCSC = Flux.Tracker.track(mul, a, b)
Flux.Tracker.@grad function mul(a::AbstractMatrix, b::SparseMatrixCSC)
  return mul(Flux.data(a),b) , Δ -> (multrans(Δ, b),nothing)
end


w = param(randn(2,3));
x = sprand(3,2,0.1);
Flux.back!(sum(w*x))
w.grad