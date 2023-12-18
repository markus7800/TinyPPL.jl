
#=
    transforms
    let x = v1, y = v2, ...
        body
    end
    to
    let x = v1
        let y = v2
            ...
                body
            ...
        end
    end
=#
function unwrap_let(expr)
    if expr isa Expr
        if expr.head == :let
            @assert length(expr.args) == 2
            bindings = expr.args[1]
            @assert bindings.head == :block || bindings.head == :(=) bindings
            body = unwrap_let(expr.args[2])
            @assert body.head == :block body.head

            if bindings.head == :(=)
                return Expr(expr.head, unwrap_let.(expr.args)...)
            end
            if bindings.head == :block
                @assert all(arg.head == :(=) for arg in bindings.args) bindings.args
                @assert length(bindings.args) > 0 bindings
                # @assert length(bindings.args) > 1 bindings.args, single assignments can alos be in block

                # println("body: ", body)
                current_let = Expr(:let, unwrap_let(bindings.args[end]), body)
                for binding in reverse(bindings.args[1:end-1])
                    # println(binding)
                    current_let = Expr(:let, unwrap_let(binding), current_let)
                end
                return current_let
            end
        else
            return Expr(expr.head, unwrap_let.(expr.args)...)
        end
    else
        return expr
    end
end

#=
    extracts variables of all let expressions
    e.g. extracts [:x, :y] from
    let x = 1,
        let y = 2
            body
        end
    end
    lets have to be unwrapped
=#
function get_let_variables(expr, vars=Symbol[])
    if expr isa Expr
        if expr.head == :let
            bindings = expr.args[1]
            @assert bindings.head == :(=)
            push!(vars, bindings.args[1])
            get_let_variables(bindings.args[2], vars)
            get_let_variables(expr.args[2], vars)
        else
            for arg in expr.args
                get_let_variables(arg, vars)
            end
        end
    end
    return vars
end

#=
    extracts all symbols that were not assigned in let
    e.g. extracts [:x, :println] from
    let x = 1
        println(y)
    end
=#
function get_free_variables(expr, free=Set{Symbol}(), bound=Symbol[])
    if expr isa Symbol
        if !(expr in bound)
            push!(free, expr)
        end
    elseif expr isa Expr
        if expr.head == :let 
            binding = expr.args[1]
            variable = binding.args[1]
            body = expr.args[2]
            @assert binding.head == :(=)
            get_free_variables(binding.args[2], free, bound)
            push!(bound, variable)
            get_free_variables(body, free, bound)
            @assert pop!(bound) == variable
        else
            for arg in expr.args
                get_free_variables(arg, free, bound)
            end
        end
    end
    return free
end

#=
    subsitutes all occurences of varriable `var` with expression `with`,
    respecting the scope of `var`.
    e.g. transform
    let x = x + 1
        println(x, y)
    end
    to
    for :x => 42 to
    let x = 42 + 1
        println(x, y)
    end
    and for :y => 42 to
    let x = x + 1
        println(x, 42)
    end
=#
function substitute(var::Symbol, with, in_expr)
    if in_expr isa Expr
        if in_expr.head == :let 
            binding = in_expr.args[1]
            @assert binding.head == :(=)
            if binding.args[1] == var
                # redefinition of variable, do not substitute in let body
                # however, old variable can still be in expression that we assign
                return Expr(:let,
                    Expr(:(=), var, substitute(var, with, binding.args[2])),
                in_expr.args[2])
            end
        end
        return Expr(in_expr.head, [substitute(var, with, arg) for arg in in_expr.args]...)
    elseif in_expr isa Symbol && in_expr == var
        return with
    else
        return in_expr
    end
end
function substitute(var_to_expr::Dict{Symbol,Any}, in_expr)
    if in_expr isa Expr
        if in_expr.head == :let 
            binding = in_expr.args[1]
            @assert binding.head == :(=)
            if haskey(var_to_expr, binding.args[1])
                var = binding.args[1]
                with_expr = var_to_expr[var]
                # redefinition of variable, do not substitute in let body
                # however, old variable can still be in expression that we assign
                new_binding = substitute(var_to_expr, binding.args[2])
                delete!(var_to_expr, var)
                new_body = substitute(var_to_expr, in_expr.args[2])
                var_to_expr[var] = with_expr
                return Expr(:let, Expr(:(=), var, new_binding), new_body)
            end
        end
        return Expr(in_expr.head, [substitute(var_to_expr, arg) for arg in in_expr.args]...)
    elseif in_expr isa Symbol && haskey(var_to_expr, in_expr)
        return var_to_expr[in_expr]
    else
        return in_expr
    end
end

#=
    Substitutes all occurences of expression `expr` with expression `with` in expression `in_expr`.
    e.g. transforms
    let X = [1,2,3,4,5]
        println(X[2,3,4])
    end
    for :(X[2,3,4]) => :(X[2:4])
    let X = [1,2,3,4,5]
        println(X[2:4])
    end
=#
function substitute_expr(expr, with, in_expr)
    if expr == in_expr
        return with
    elseif in_expr isa Expr
        return Expr(in_expr.head, [substitute_expr(expr, with, arg) for arg in in_expr.args]...)
    else
        return in_expr
    end
end

#=
    Removes if expression with condition `true`.
    And replaces `true && phi` with `phi`.
=#
function simplify_if(expr)
    if expr isa Expr
        if expr.head == :if
            if expr.args[1] == true
                return expr.args[2]
            end
        end
        if expr.head == :&&
            if expr.args[1] == true
                return expr.args[2]
            end
        end
        return Expr(expr.head, simplify_if.(expr.args)...)
    else
        return expr
    end
end

function repeatf_symbolic(n, f, x)
    v = gensym(:v)
    block_args = [:($v = $f($x))]
    for _ in 1:(n-1)
        push!(block_args, :($v = $f($v)))
    end
    return unwrap_let(Expr(:let, Expr(:block, block_args...), Expr(:block, v)))
end

#=
    extracts names of function calls
=#
function get_call_names(expr, names=Symbol[])
    if expr isa Expr
        if expr.head == :call
            name = expr.args[1]
            if name isa Symbol
                push!(names, name)
            elseif name isa Expr && name.head == :.
                @assert name.args[2] isa QuoteNode && name.args[2].value isa Symbol (name.args[2], typeof(name.args[2]))
                push!(names, name.args[2].value)
            end
        end
        for arg in expr.args
            get_call_names(arg, names)
        end
    end
    return names
end