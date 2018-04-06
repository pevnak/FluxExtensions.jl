function learn(loss,opt,data,cb = ()->();cbreak = 100)
	fVal = 0.0
	t = zeros(2)
	tic()
	for (i,x) in enumerate(data)
		t[1] += toq()
		tic()
		l = loss(x...)
		Flux.Tracker.back!(l)
		opt()
		t[2] += toq()
		fVal += Flux.data(l)
		if mod(i,cbreak) == 0
			println(@sprintf("%d: error = %g load time = %.2fs compute time = %.2fs",i,fVal/cbreak,t[1],t[2]))
			fVal = 0.0
		end
		tic()
	end
	toq();
end