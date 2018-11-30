using Optim

function vector2values!(dst, θ)
    offset = 1
    for m in dst 
        copy!(m.data, θ[offset:offset+length(m)-1])
        offset += length(m)
    end
end

function grad2vector!(w, src)
    offset = 1
    for m in src 
        copy!(view(w,offset:offset+length(m)-1), m.grad)
        # fill!(m.grad,0)
        offset += length(m)
    end
end

function lbfgs(optfun, m, x; testgradient::Bool = false, options = Optim.Options(g_tol = 1e-8,
                             iterations = 50000,
                             f_calls_limit = 10000,
                             store_trace = false,
                             show_trace = true))
    function f(θ, m, ps, x)
        vector2values!(ps, θ)
        optfun(m,x)
    end

    function g!(w, θ, m, ps, x)
        vector2values!(ps, θ)
        zerograds!(ps)
        fVal = optfun(m, x);
        Flux.back!(fVal)
        grad2vector!(w, ps)
        Flux.data(fVal)
    end

    #test the gradient:
    ps = params(m)
    θ = randn(sum(length.(ps)))
    if testgradient
        w = zeros(length(θ))
        g!(w, θ)
        δ = maximum(abs.(Flux.Tracker.ngradient(f,θ)[1] .- w))
        δ < 1e-6 && error("gradient test failed, difference ",δ)
    end

    mf = freeze(m)
    optimize(θ -> f(θ, mf, ps, x) , (w,θ) ->  g!(w, θ, m, ps, x), θ, LBFGS(), options)
end