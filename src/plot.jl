function plotheatmap(x,f)
	xl = linspace(minimum(x[1,:]),maximum(x[1,:]),100)
	yl = linspace(minimum(x[2,:]),maximum(x[2,:]),100)
	xx = hcat([[i,j] for i in xl for j in yl]...);

	z =  reshape(f(xx),length(xl),length(yl));
	# z = reshape(model(xx).data,length(xl),length(yl))
	heatmap(xl,yl,z)
end

