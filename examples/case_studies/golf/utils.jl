# Plotting helpers for golf.jl.

using Plots
import FileIO
include("../../common/plotting.jl")

const GOLFER_IMAGE = FileIO.load(joinpath(@__DIR__, "resources", "golf_player.jpg"))
const POND_IMAGE = FileIO.load(joinpath(@__DIR__, "resources", "golf_pond.png"))
const HOLE_IMAGE = FileIO.load(joinpath(@__DIR__, "resources", "golf_hole.png"))

# Draws an image with its centre at (x, y) and the given height in data coordinates (keeping its aspect ratio).
function draw_image!(p, image, x, y, height)
    width = height * size(image, 2) / size(image, 1)
    plot!(p, range(x - width / 2, x + width / 2, length=size(image, 2)), range(y - height / 2, y + height / 2, length=size(image, 1)),
        reverse(image, dims=1), yflip=false)
end

function plot_golfcourse(course)
    xlims = (min(course.player_x, course.hole_x, course.pond_x) - 2, max(course.player_x, course.hole_x, course.pond_x) + 4)
    p = plot(xlims=xlims, ylims=(-0.5, 7), aspect_ratio=:equal, legend=false, size=(900, 500), grid=false)
    draw_image!(p, HOLE_IMAGE, course.hole_x + 0.18, 0.5, 1.0)
    draw_image!(p, POND_IMAGE, course.pond_x, 0.1, 0.8)
    draw_image!(p, GOLFER_IMAGE, course.player_x - 0.33, 0.75, 1.2)
    return p
end

# Trajectories of the strikes in the traces (transparent), and a histogram of where the balls end up.
function plot_traces(course, traces)
    p = plot_golfcourse(course)
    end_positions = zeros(length(traces))
    for t in 1:length(traces)
        strike = get_strike(traces[:angle][t], traces[:power][t])
        trajectory, end_positions[t] = get_trajectory(course, [course.player_x, 0.], strike, traces[:wind][t])
        plot!(p, first.(trajectory), last.(trajectory), color=:black, alpha=0.02)
    end
    histogram!(p, end_positions, normalize=:pdf, bins=range(course.player_x, course.hole_x + 4, length=100), alpha=0.75, linewidth=0)
    return p
end

# A strike with angle 58.1° and power 0.66 in a tail wind of 0.03, which we observe in golf.jl.
function get_observed_trajectory(course)
    trajectory, _ = get_trajectory(course, [0., 0.], get_strike(deg2rad(58.1), 0.66), 0.03)
    return hcat(trajectory...)
end
