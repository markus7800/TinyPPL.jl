using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
import PyPlot
using PyCall
import ForwardDiff

plt = pyimport("matplotlib.pyplot");
plt_image = pyimport("matplotlib.image");
plt_box = pyimport("matplotlib.offsetbox");
plt_cm = pyimport("matplotlib.cm");
np = pyimport("numpy");

include("utils.jl");

# The Power of Probabilistic Programming

## Scenario:
# 1. **We have existing physics simulation.**
# 2. **We have some unknown parameters of the simulation.**
# 3. **We want to use Bayesian statistics to infer those unknowns.**


## The Environment:
# A golf course which has
# - a player position
# - a target hole position
# - a pond position

struct GolfCourse
    player_x::Float64
    hole_x::Float64
    pond_x::Float64
    function GolfCourse(;player_x::Float64, hole_x::Float64, pond_x::Float64)
        return new(player_x, hole_x, pond_x)
    end
end

pond_x0(course::GolfCourse) = course.pond_x - 0.95
pond_x1(course::GolfCourse) = course.pond_x + 0.9 #+ 0.7

function inpond(course::GolfCourse, x)
    return pond_x0(course) < x && x < pond_x1(course)
end;

## The simulation:
# Simple classical mechanics simulation with
# - initial velocity = `strike`
# - acceleration = `[drag + wind, gravity]`, where `drag = -0.1` and `gravity = -1`  

# The ball stays in pond if it hits the pond.  
# Otherwise, we allow the ball to bounce 3 times.

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
    
    # if position[2] != 0.
    #     _strike = ForwardDiff.value.(strike)
    #     _wind = ForwardDiff.value(wind)
    #     _end_x_position = ForwardDiff.value(position[2])
    #     @warn "Simulation did not finish" _strike _wind _end_x_position
    # end

    return trajectory, end_x_position
end;

function get_strike(angle::Float, power::Float) where Float <: Real
    # angle in [0,pi/2]
    # power in [0.25,1.]
    r = (power - 0.5) * 2 # in [-0.5,1.]
    p = r + 2.75
    return [cos(angle) * p, sin(angle) * p]
end

# Plotting some trajectories

course = GolfCourse(player_x=0., hole_x=8., pond_x=5.5)
fig, ax = plot_golfcourse(course)
trajectory, end_x_position = get_trajectory(course, [0.,0.], get_strike(deg2rad(30),0.6), 0.)
trajectory, end_x_position = get_trajectory(course, [0.,0.], get_strike(deg2rad(65),0.6), 0.)
trajectory, end_x_position = get_trajectory(course, [0.,0.], get_strike(deg2rad(60),0.9), 0.)
#trajectory, end_x_position = get_trajectory(course, [0.,0.], get_strike(deg2rad(60),0.9), -0.1)
trajectory = hcat(trajectory...)
ax.scatter(trajectory[1,:], trajectory[2,:]);
# ax.vlines([pond_x0(course),pond_x1(course)],-1,1)
plt.gcf()


## Modelling the unknowns:

# - Golfer has `goal_angle` and `goal_power`
# - But depending on his `skill_level` he may deviate from the desired strike.
# - This is expressed by Normal distribution with deviations depending on the `skill_level`:

# angle ~ Normal(goal_angle, angle_deviation)
# power ~ Normal(goal_power, power_deviation)

struct Golfer
    skill_level::Int # (1,2,3)
    function Golfer(;skill_level::Int)
        return new(skill_level)
    end
end

@ppl static function strike(golfer::Golfer, goal_angle::Real, goal_power::Real)
    # @assert golfer.skill_level in (1,2,3)
        
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
end;

# In the `play_golf` model, we simply wrap our physics simulation.
# - `goal_angle` can range from 0 to 85 degrees.
# - `goal_power` can range from 0.25 to 1.00 percent.
# - We instantiate a golfer with set `skill_level` and sample a strike.
# - The `wind` is also unknown.
# - We simulate the trajectory and record it's end position.

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
        # hitting the golf holf perfectly has probability 0.
        # we have to allow some margin 0.1
        end_position ~ Normal(end_x_position, 0.1)
    else
        # just for tracking the end position
        end_position ~ Dirac(end_x_position)
    end

    return end_x_position
end;

# To get a feel for the model, we will sample from the prior distribution.

Random.seed!(0);
traces = sample_from_prior(
    play_golf, (course, Golfer(skill_level=3), false),
    Observations(:goal_angle => deg2rad(55), :goal_power => 0.7),
    1000);
plot_traces(course, traces);
plt.gcf()


## Inverting the Model:

# Up until now we simply generated random golf strikes.
# But we can also use probabilistic programming to infer unkown values.
# In fact, we can use probabilistic programming to answer following question:

### GIVEN that we hit the golf hole, what are angle and power?

Random.seed!(0);
traces_LW, lps = likelihood_weighting(
    play_golf, (course, Golfer(skill_level=3), true),
    Observations(:end_position => course.hole_x),
    500_000);
weights_LW = exp.(lps)
goal_angle = rad2deg.(traces_LW[:goal_angle]);
goal_power = traces_LW[:goal_power];

# Visualising the results allows us to interpret the joint posterior distribution of angle and power.
# That means we do not only get one solution, but an entire distribution over angle and power that works well.

plt.figure()
plt.hexbin(goal_angle, goal_power, C=weights_LW, reduce_C_function=np.sum, gridsize=50);
plt.xlabel("angle"); plt.ylabel("power");
plt.gcf()

# We can also visualise the trajectories that belong to these solution values.

Random.seed!(0)
ixs = rand(Categorical(weights_LW),1000)
plot_traces(course, subset(traces_LW, ixs));
plt.gcf()

# Maybe it is also interesting to see, which are the angle and power values that are the most likely to hit the goal.

# Estimates the MAP by finding the highest weighted bin of the posterior histogram.
counts, x, y = np.histogram2d(goal_angle, goal_power, weights=weights_LW, bins=100);
i, j = Tuple(argmax(counts))
map_angle = 0.5 * (x[i+1] + x[i])
map_power = 0.5 * (y[j+1] + y[j])
map_angle, map_power

Random.seed!(0);
traces_3 = sample_from_prior(
    play_golf, (course, Golfer(skill_level=3), false),
    Observations(:goal_angle => deg2rad(map_angle), :goal_power => map_power),
    1000);
plot_traces(course, traces_3);
plt.gcf()

# Results look fine so far, but we always assume that we have professional golfer with `skill_level = 3`.

### What happens if we let a less experienced golfer give it a shot?

Random.seed!(0);
traces_1 = sample_from_prior(
    play_golf, (course, Golfer(skill_level=1), false),
    Observations(:goal_angle => deg2rad(map_angle), :goal_power => map_power),
    1000);
plot_traces(course, traces_1);
plt.gcf()

#### What is the probability of hitting the pond?

mean(map(x -> inpond(course, x), vec(traces_1[:end_position])))


Random.seed!(0);
traces_2 = sample_from_prior(
    play_golf, (course, Golfer(skill_level=1), false),
    Observations(:goal_angle => deg2rad(map_angle), :goal_power => map_power),
    1000);
plot_traces(course, traces_2);
plt.gcf()


println("Prob. of hitting pond skill_level = 3: ", mean(map(x -> inpond(course, x), vec(traces_3[:end_position]))))
println("Prob. of hitting pond skill_level = 2: ", mean(map(x -> inpond(course, x), vec(traces_2[:end_position]))))
println("Prob. of hitting pond skill_level = 1: ", mean(map(x -> inpond(course, x), vec(traces_1[:end_position]))))

### Golfer with skill level 2, seems to be good choice, but we observe bad trajectory!?

observed_trajectory = get_observed_trajectory(course)
fig, ax = plot_golfcourse(course)
ax.plot(observed_trajectory[1,:], observed_trajectory[2,:]);
plt.gcf()



Random.seed!(0);
result, lps = likelihood_weighting(
    play_golf, (course, Golfer(skill_level=2), true),
    Observations(
        :goal_angle => deg2rad(map_angle),
        :goal_power => map_power,
        :end_position => observed_trajectory[1,end]),
    100_000);
weights= exp.(lps);


fig, axs = plt.subplots(1,3,figsize=(15,5));
axs[1].hist(result[:power], weights=weights, bins=25, label="simulated power");
axs[1].vlines([map_power], 0, 0.5, color="tab:orange", label="goal power");
axs[1].legend();
axs[2].hist(rad2deg.(result[:angle]), weights=weights, bins=25, label="simulated angle");
axs[2].vlines([map_angle], 0, 0.5, color="tab:orange", label="goal angle");
axs[2].legend();
axs[3].hist(result[:wind], weights=weights, bins=25, label="simulated wind");
axs[3].vlines([0.], 0, 0.5, color="tab:orange", label="expected wind");
axs[3].legend();
plt.gcf()


## Gradient-Based Methods

# Since the simulation is differentiable with respect to angle and power,  
# we can use more sophisticated inference algorithms that leverage automatic differentiation.

inverse_model = (play_golf, (course, Golfer(skill_level=3), true), Observations(:end_position => course.hole_x))
get_address_to_ix(inverse_model...)

initial_params = [deg2rad(map_angle),map_power,deg2rad(map_angle),map_power,0.1];

Random.seed!(0);
traces_HMC = hmc(inverse_model..., 10_000, 10, 0.003; ad_backend=:forwarddiff, x_initial=initial_params);



function plot_HMC_result(traces_HMC)
    fig, axs = plt.subplots(1,2,figsize=(12,4))

    goal_angle_IS = rad2deg.(traces_LW[:goal_angle])
    goal_power_IS = traces_LW[:goal_power]
    axs[1].hexbin(goal_angle_IS, goal_power_IS, C=weights_LW, reduce_C_function=np.sum, gridsize=50);
    axs[1].set_xlabel("angle"); axs[1].set_ylabel("power");
    axs[1].set_ylim((0.25,1.)); axs[1].set_xlim((5, 85))

    goal_angle_HMC = rad2deg.(traces_HMC[:goal_angle])
    goal_power_HMC = traces_HMC[:goal_power];

    axs[2].hexbin(goal_angle_HMC, goal_power_HMC, gridsize=50);
    axs[2].set_ylim((0.25,1.)); axs[2].set_xlim((5, 85))
    axs[2].set_xlabel("angle"); axs[2].set_ylabel("power");
end
plot_HMC_result(traces_HMC);
plt.gcf()