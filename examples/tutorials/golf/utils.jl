function normalise(logprobs::AbstractVector{<:Real})
    m = maximum(logprobs)
    l = m + log(sum(exp, logprobs .- m))
    return logprobs .- l
end


function plot_golfcourse(golf_course; fig_height::Int=8)
    positions = [golf_course.player_x, golf_course.hole_x]
    if !isnothing(golf_course.pond_x)
        push!(positions, golf_course.pond_x)
    end
    xlims = (minimum(positions)-2, maximum(positions)+4)
    ylims = (-0.5, 7)#(xlims[1]-xlims[0])/2)

    plt.close();
    fig, ax = plt.subplots(figsize = ((xlims[2]-xlims[1])/(ylims[2]-ylims[1])*fig_height, fig_height))
    ax.set_aspect("equal")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    scale = fig_height/8

    golfer = plt_image.imread("examples/tutorials/golf/golf_resources/golf_player.jpg")
    pond = plt_image.imread("examples/tutorials/golf/golf_resources/golf_pond.png")
    hole = plt_image.imread("examples/tutorials/golf/golf_resources/golf_hole.png")
    golfer_box = plt_box.OffsetImage(golfer, zoom=0.20*scale)
    pond_box = plt_box.OffsetImage(pond, zoom=0.22*scale)
    hole_box = plt_box.OffsetImage(hole, zoom=0.22*scale)


    ab = plt_box.AnnotationBbox(hole_box, (golf_course.hole_x+0.18,0.4), frameon=false, zorder=0)
    ax.add_artist(ab)

    ab = plt_box.AnnotationBbox(golfer_box, (golf_course.player_x-0.33,0.75), frameon=false, zorder=0)
    ax.add_artist(ab)

    if !isnothing(golf_course.pond_x)
        ab = plt_box.AnnotationBbox(pond_box, (golf_course.pond_x,0.3), frameon=false, zorder=0)
        ax.add_artist(ab)
    end
    return fig, ax
end

# Plotting the golf strikes
function plot_traces(course, result)
    plt.close()
    fig, ax = plot_golfcourse(course, fig_height=8)
    end_x_positions = Vector{Float64}(undef,length(result))
    for t in 1:length(result)
        angle = result[:angle][t]
        power = result[:power][t]
        wind = result[:wind][t]
        golfer_strike = get_strike(angle, power)
        trajectory, end_x_positions[t] = get_trajectory(course, [course.player_x,0.], golfer_strike, wind)
        ax.plot([pos[1] for pos in trajectory], [pos[2] for pos in trajectory], color="black", alpha=0.01)
    end
    
    bins = LinRange(course.player_x,course.hole_x+4,100)
    plt.hist(end_x_positions, density=true, bins=bins, alpha=0.75)
end

function get_observed_trajectory(course)
    trajectory, end_x_position = get_trajectory(course, [0.,0.], get_strike(deg2rad(58.1),0.66), 0.03)
    trajectory = hcat(trajectory...)
    return trajectory
end