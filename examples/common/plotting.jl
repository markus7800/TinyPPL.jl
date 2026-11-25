using Plots

const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")

# In the REPL the plot is displayed, when run as a script it is saved to examples/output/<name>.png.
function show_plot(p, name::String)
    if isinteractive()
        display(p)
    else
        mkpath(OUTPUT_DIR)
        path = joinpath(OUTPUT_DIR, name * ".png")
        savefig(p, path)
        println("Saved plot to ", normpath(path))
    end
    return p
end
