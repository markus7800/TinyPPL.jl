# The power of probabilistic programming: inferring the unknowns of an existing physics simulation of golf strikes.
#
# 1. We have an existing physics simulation of a golf ball.
# 2. Some of its parameters are unknown: the goal angle and power of the strike, the deviation of the golfer from it,
#    and the wind.
# 3. We use Bayesian inference to infer these unknowns from observations, e.g. that the ball ended up in the hole.
#
# Run: julia --project=examples examples/case_studies/golf/golf.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("utils.jl")

# ===== The environment =====
# A golf course with the positions of the player, the target hole and a pond.
struct GolfCourse
    player_x::Float64
    hole_x::Float64
    pond_x::Float64
    function GolfCourse(;player_x::Float64, hole_x::Float64, pond_x::Float64)
        return new(player_x, hole_x, pond_x)
    end
end

pond_x0(course::GolfCourse) = course.pond_x - 0.95
pond_x1(course::GolfCourse) = course.pond_x + 0.9

function inpond(course::GolfCourse, x)
    return pond_x0(course) < x && x < pond_x1(course)
end

# ===== The simulation =====
# Simple classical mechanics with initial velocity `strike` and acceleration [drag + wind, gravity],
# where drag = -0.1 and gravity = -1. The ball stays in the pond if it lands there, otherwise it bounces twice.

function get_trajectory(golf_course::GolfCourse, start::Vector{Float64}, strike::Vector{Float}, wind::Float, dt::Float64=0.1) where Float <: Real
    position = start
    trajectory::Vector{Vector{Float}} = Vector{Float}[position]
    gravity = -1.
    drag = -0.1
    acceleration = Float[wind, gravity]
    velocity = strike
    end_x_position = 0.
    bounce = 0
    n_bounce = 2
    
    for i in 1:150 # upper bound on iterations
        acceleration[1] = wind + drag * sign(velocity[1])
        velocity = velocity + dt * acceleration
        position = position + dt * velocity
        if position[2] < 0.0
            # compute intersection with x-axis
            p = position
            q = trajectory[end]
            k = (q[1] - p[1]) / (q[2] - p[2])
            inter_section_x = p[1] - k * p[2]
            
            if bounce < n_bounce && !inpond(golf_course, inter_section_x)
                # if ball is not in pond, bounce
                position[2] = -position[2]
                velocity[1] = velocity[1] / 2.
                velocity[2] = -velocity[2] / 2.
                bounce += 1
            else
                # if ball is in pond or has bounced 3 times stop
                end_x_position = inter_section_x
                position[1] = inter_section_x
                position[2] = 0.0
            end
        end
        push!(trajectory, position)
        
        if position[2] == 0.
            break
        end
    end
    
    return trajectory, end_x_position
end

function get_strike(angle::Float, power::Float) where Float <: Real
    # angle in [0,pi/2]
    # power in [0.25,1.]
    r = (power - 0.5) * 2 # in [-0.5,1.]
    p = r + 2.75
    return [cos(angle) * p, sin(angle) * p]
end

course = GolfCourse(player_x=0., hole_x=8., pond_x=5.5)
p = plot_golfcourse(course)
for (angle, power) in [(30, 0.6), (65, 0.6), (60, 0.9)]
    trajectory, _ = get_trajectory(course, [0., 0.], get_strike(deg2rad(angle), power), 0.)
    scatter!(p, first.(trajectory), last.(trajectory), markersize=2, markerstrokewidth=0)
end
show_plot(p, "golf_example_trajectories")


# ===== The model =====
# The golfer has a goal angle and power, but depending on their skill level (1, 2 or 3),
# the strike deviates from it: angle ~ Normal(goal_angle, deviation), power ~ Normal(goal_power, deviation).
struct Golfer
    skill_level::Int # (1,2,3)
    function Golfer(;skill_level::Int)
        return new(skill_level)
    end
end

@ppl static function strike(golfer::Golfer, goal_angle::Real, goal_power::Real)
    if golfer.skill_level == 1
        angle_deviation = 0.05
        power_deviation = 0.05
    elseif golfer.skill_level == 2
        angle_deviation = 0.0125
        power_deviation = 0.0125
    elseif golfer.skill_level == 3
        angle_deviation = 0.00625
        power_deviation = 0.00625
    end
    
    angle ~ Normal(goal_angle, angle_deviation)
    power ~ Normal(goal_power, power_deviation)
    
    return get_strike(angle, power)
end

# The play_golf model wraps the physics simulation:
# - goal_angle ranges from 5 to 85 degrees and goal_power from 0.25 to 1.
# - The golfer strikes with some deviation from the goal, depending on their skill level.
# - The wind is unknown.
# - We simulate the trajectory and record its end position.

@ppl static function play_golf(
    course::GolfCourse,
    golfer::Golfer,
    inverse_problem::Bool
)

    goal_angle ~ Uniform(deg2rad(5), deg2rad(85))
    goal_power ~ Uniform(0.25, 1.0)

    golfer_strike = @subppl strike(golfer, goal_angle, goal_power)

    wind ~ Normal(0., 0.015)
    _, end_x_position = get_trajectory(course, Float64[course.player_x,0.], golfer_strike, wind)


    if inverse_problem
        # observing the exact end position has probability 0, so we allow some noise
        end_position ~ Normal(end_x_position, 0.1)
    else
        # just for tracking the end position
        end_position ~ Dirac(end_x_position)
    end

    return end_x_position
end


# Sampling from the prior for fixed goal angle and power shows the possible strikes of a professional golfer.
Random.seed!(0)
traces = sample_from_prior(play_golf, (course, Golfer(skill_level=3), false), Observations(:goal_angle => deg2rad(55), :goal_power => 0.7), 1000)
show_plot(plot_traces(course, traces), "golf_prior")


# ===== Inverting the model =====
# Given that the ball ended up in the hole, what are the goal angle and power?
# Likelihood weighting gives a joint posterior distribution over angle and power: not one solution, but all that work.
Random.seed!(0)
traces_LW, lps = likelihood_weighting(play_golf, (course, Golfer(skill_level=3), true), Observations(:end_position => course.hole_x), 500_000)
W_LW = exp.(lps)
goal_angle = rad2deg.(traces_LW[:goal_angle])
goal_power = traces_LW[:goal_power]
p_LW = histogram2d(goal_angle, goal_power, weights=W_LW, bins=50, xlabel="goal angle", ylabel="goal power", xlims=(5, 85), ylims=(0.25, 1), title="likelihood weighting", color=:Blues)
show_plot(p_LW, "golf_posterior_angle_power")

# The trajectories of posterior samples, drawn proportionally to their weights.
Random.seed!(0)
ixs = rand(Categorical(W_LW), 1000)
show_plot(plot_traces(course, subset(traces_LW, ixs)), "golf_posterior_trajectories")

# The most likely goal angle and power, estimated as the centre of the highest bin of a weighted 2D histogram.
function weighted_mode(xs, ys, W; bins=100)
    x_edges = range(minimum(xs), maximum(xs), length=bins+1)
    y_edges = range(minimum(ys), maximum(ys), length=bins+1)
    counts = zeros(bins, bins)
    for (x, y, w) in zip(xs, ys, W)
        counts[clamp(searchsortedlast(x_edges, x), 1, bins), clamp(searchsortedlast(y_edges, y), 1, bins)] += w
    end
    i, j = Tuple(argmax(counts))
    return (x_edges[i] + x_edges[i+1]) / 2, (y_edges[j] + y_edges[j+1]) / 2
end
map_angle, map_power = weighted_mode(goal_angle, goal_power, W_LW)
println("most likely goal angle ≈ ", round(map_angle, digits=1), "°, goal power ≈ ", round(map_power, digits=3))


# ===== Golfers with different skill levels =====
# So far we assumed a professional golfer (skill level 3). What happens if less experienced golfers aim for the same strike?
plots = []
for skill_level in [3, 2, 1]
    Random.seed!(0)
    local traces = sample_from_prior(play_golf, (course, Golfer(skill_level=skill_level), false),
        Observations(:goal_angle => deg2rad(map_angle), :goal_power => map_power), 1000)
    println("skill level ", skill_level, ": P(ball in pond) ≈ ", mean(x -> inpond(course, x), traces[:end_position]))
    push!(plots, plot(plot_traces(course, traces), title="skill level $skill_level"))
end
show_plot(plot(plots..., layout=(3, 1), size=(900, 1200)), "golf_skill_levels")


# ===== Explaining an unexpected observation =====
# A golfer with skill level 2 aims for the most likely angle and power, but the ball lands far behind the hole.
# What went wrong: the angle, the power or the wind?
observed_trajectory = get_observed_trajectory(course)
p = plot_golfcourse(course)
plot!(p, observed_trajectory[1, :], observed_trajectory[2, :], linewidth=2)
show_plot(p, "golf_observed_trajectory")

Random.seed!(0)
traces, lps = likelihood_weighting(play_golf, (course, Golfer(skill_level=2), true),
    Observations(:goal_angle => deg2rad(map_angle), :goal_power => map_power, :end_position => observed_trajectory[1, end]), 100_000)
W = exp.(lps)
println("posterior means: power ≈ ", round(W'traces[:power], digits=3), " (goal ", round(map_power, digits=3), "), angle ≈ ",
    round(rad2deg(W'traces[:angle]), digits=1), "° (goal ", round(map_angle, digits=1), "°), wind ≈ ", round(W'traces[:wind], digits=4), " (prior mean 0)")
p = plot(layout=(1, 3), size=(1200, 350), legend=:topleft)
histogram!(p[1], traces[:power], weights=W, normalize=:pdf, bins=25, label="power", linewidth=0); vline!(p[1], [map_power], label="goal power", linewidth=2)
histogram!(p[2], rad2deg.(traces[:angle]), weights=W, normalize=:pdf, bins=25, label="angle", linewidth=0); vline!(p[2], [map_angle], label="goal angle", linewidth=2)
histogram!(p[3], traces[:wind], weights=W, normalize=:pdf, bins=25, label="wind", linewidth=0); vline!(p[3], [0.], label="prior mean", linewidth=2)
show_plot(p, "golf_observed_trajectory_posterior")


# ===== Gradient-based inference =====
# The simulation is differentiable with respect to angle, power and wind, so we can use HMC with automatic differentiation.
# We start the chain at the most likely goal angle and power (the order of the initial values follows the addresses).
inverse_model = (play_golf, (course, Golfer(skill_level=3), true), Observations(:end_position => course.hole_x))
addresses_to_ix = get_address_to_ix(inverse_model...)
x_initial = zeros(length(addresses_to_ix))
for (addr, value) in [:goal_angle => deg2rad(map_angle), :goal_power => map_power, :angle => deg2rad(map_angle), :power => map_power, :wind => 0.]
    x_initial[addresses_to_ix[addr]] = value
end

Random.seed!(0)
traces_HMC = hmc(inverse_model..., 10_000, 10, 0.003; ad_backend=:forwarddiff, x_initial=x_initial)
p_HMC = histogram2d(rad2deg.(traces_HMC[:goal_angle]), traces_HMC[:goal_power], bins=50, xlabel="goal angle", ylabel="goal power", xlims=(5, 85), ylims=(0.25, 1), title="HMC", color=:Blues)
show_plot(plot(p_LW, p_HMC, layout=(1, 2), size=(1000, 400)), "golf_HMC")
