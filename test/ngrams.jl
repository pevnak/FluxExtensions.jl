using FluxExtensions: NGramIterator, ngrams, string2ngrams
@testset "ngrams" begin
	all(collect(NGramIterator(codeunits("hello"), 3, 257)) .== ngrams("hello", 3, 257))

	A = randn(4, 10)
	@test all(mul(A , ["hello","hello"]) .≈ A*string2ngrams(["hello","hello"], 3, size(A, 2)))
	A = randn(5,2)
	@test all(multrans(A , ["hello","hello"]) .≈ A*transpose(string2ngrams(["hello","hello"], 3, size(A, 2))))
end