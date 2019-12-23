using Statistics
using Profile

@enum BinInit begin
    LightSpeed  # set limits of initial bin +- 3.1e8 m/s
    MinMaxVel  # set limits of initial bin to min/max velocities of particles in question
end

@enum BinSplit begin
    MiddleSplit  # split bin among middle
    MeanVelSplit  # split bin along mean velocity in bin
end

@enum BinRefinement begin
    Greedy  # refine the first bin that is suitable
    Heaviest  # refine the heaviest bin
end


struct Constants
    k::Float64
end


constants = Constants(1.38064852e-23)


mutable struct Particle
    w::Float64
    v::Array{Float64,1}
    octant::Int
end

mutable struct Bin
    w::Float64  # total weight in bin
    nparticles::Int  # total number of particles in bin
    v::Array{Float64,1}  # velocity of bin
    c::Array{Float64,1}  # second moment in bin
    
    v_lo::Array{Float64,1}  # min velocity of bin
    v_hi::Array{Float64,1}  # max velocity of bin
    v_split::Array{Float64,1}  # velocities along which octant splitting is performed
    
    index_start::Int
    index_end::Int # indices of particle array - start, end of particles belonging to this bin
    
    level::Int
end


function maxwellian(T, mass)
    # sample 3-D velocity from Maxwellian
    return sqrt(constants.k * T / mass) * randn((3,))
end


function compute_octant!(particle, v_split)
    # compute index of octree bin particle is in
    particle.octant = 1
    if particle.v[1] > v_split[1]
        particle.octant += 1
    end
    if particle.v[2] > v_split[2]
        particle.octant += 2
    end
    if particle.v[3] > v_split[3]
        particle.octant += 4
    end
end


function compute_bin_values!(particle_list, bin)
    # compute density, velocity and 2nd order moments in bin
    bin.w = 0.0
    bin.nparticles = bin.index_end - bin.index_start + 1
    bin.v = [0.0, 0.0, 0.0]
    bin.c = [0.0, 0.0, 0.0]
    
    for i=bin.index_start:bin.index_end
        bin.w += particle_list[i].w
        bin.v += particle_list[i].w .* particle_list[i].v
    end
    
    if (bin.w > 0.0)
        bin.v = bin.v / bin.w
    end
    
    for i=bin.index_start:bin.index_end
        bin.c += particle_list[i].w * ((particle_list[i].v .- bin.v).^2)
    end
    
    if (bin.w > 0.0)
        bin.c = sqrt.(bin.c / bin.w)
    end
end


function compute_bin_weight_and_split!(particle_list, bin, split_type)
    # when refining, we just need to keep track of the weight and number of particles in a bin
    # also computes the velocities to split the bin along
    
    bin.w = 0.0
    bin.nparticles = bin.index_end - bin.index_start + 1
    bin.v = [0.0, 0.0, 0.0]
    
    for i=bin.index_start:bin.index_end
        bin.w += particle_list[i].w
        bin.v += particle_list[i].w .* particle_list[i].v
    end
    
    if ((split_type == MiddleSplit) || (bin.w <= 0.0))
        bin.v_split = (bin.v_lo + bin.v_hi) / 2
    else
        bin.v_split = bin.v / bin.w
    end
end


function refine_bin!(particle_list, bins_list, bin_id, n_bins, split_type)
    # in order to refine a bin, we need to keep particle sorted
    # by octant
    # we use a counting sort
    bin_counts = [0, 0, 0, 0, 0, 0, 0, 0]
    
    for particle in particle_list[bins_list[bin_id].index_start:bins_list[bin_id].index_end]
        compute_octant!(particle, bins_list[bin_id].v_split)
    end

    # a better/faster version would operate on lists of pointers to particles, not particle structs themselves
    particles_in_bin = particle_list[bins_list[bin_id].index_start:bins_list[bin_id].index_end]
    
    for particle in particle_list[bins_list[bin_id].index_start:bins_list[bin_id].index_end]
        bin_counts[particle.octant] += 1
    end
    
    bin_counts_copy = bin_counts[:]
    
    for i=2:8
        bin_counts[i] += bin_counts[i-1]
    end
    
    bin_counts_acc_copy = bin_counts[:]
    
    for particle in particle_list[bins_list[bin_id].index_start:bins_list[bin_id].index_end]
        particles_in_bin[bin_counts[particle.octant]] = particle
        bin_counts[particle.octant] -= 1
    end

    particle_list[bins_list[bin_id].index_start:bins_list[bin_id].index_end] = particles_in_bin[:]

    new_bins = 0
    
    for i=1:8
        if bin_counts_copy[i] > 0
            new_bins += 1
            
            
            # set new bin limits, split along z axis
            if i<=4
                vz_min = bins_list[bin_id].v_lo[3]
                vz_max = bins_list[bin_id].v_split[3]
            else
                vz_min = bins_list[bin_id].v_split[3]
                vz_max = bins_list[bin_id].v_hi[3]
            end
                
            
            # set new bin limits, split along y axis
            if i%2 == 0
                vy_min = bins_list[bin_id].v_split[2]
                vy_max = bins_list[bin_id].v_hi[2]
            else
                # lower vy values
                vy_min = bins_list[bin_id].v_lo[2]
                vy_max = bins_list[bin_id].v_split[2]
            end
                    
            if ((i==3) || (i==4) || (i==7) || (i==8))
                vx_min = bins_list[bin_id].v_split[1]
                vx_max = bins_list[bin_id].v_hi[1]
            else
                vx_min = bins_list[bin_id].v_lo[1]
                vx_max = bins_list[bin_id].v_split[1]
            end

                        
            
            if i==1
                istart = bins_list[bin_id].index_start
            else
                istart = bin_counts_acc_copy[i-1] + bins_list[bin_id].index_start
            end
            iend = bin_counts_acc_copy[i] + bins_list[bin_id].index_start - 1
            
            # split are zeros for now, we will compute them
            push!(bins_list, Bin(0.0, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                 [vx_min, vy_min, vz_min], [vx_max, vy_max, vz_max], [0.0, 0.0, 0.0],
                                 istart, iend,
                                 bins_list[bin_id].level+1))

            compute_bin_weight_and_split!(particle_list, bins_list[n_bins+new_bins], split_type)
        end
    end
    
    bins_list[bin_id:n_bins + new_bins-1] = bins_list[bin_id+1:n_bins + new_bins]
    
    resize!(bins_list, n_bins + new_bins - 1)
    # shift bins to left to overwrite bin
    return n_bins + new_bins - 1
end


function init_bin(particle_list, bin_init, split_type)
    # create initial bin and set initial velocity splitting values
    vx_min = 3.1e8
    vx_max = -3.1e8
    vy_min = 3.1e8
    vy_max = -3.1e8
    vz_min = 3.1e8
    vz_max = -3.1e8
    
    if bin_init == LightSpeed
        print("initial bounds set to +- 1.03c\n")
    else
        print("initial bounds set to min-max velocities\n")
        
        for particle in particle_list
            if particle.v[1] < vx_min
                vx_min = particle.v[1]
            elseif particle.v[1] > vx_max
                vx_max = particle.v[1]
            end

            if particle.v[2] < vy_min
                vy_min = particle.v[2]
            elseif particle.v[2] > vy_max
                vy_max = particle.v[2]
            end

            if particle.v[3] < vz_min
                vz_min = particle.v[3]
            elseif particle.v[3] > vz_max
                vz_max = particle.v[3]
            end
        end
    end
    
                
    # compute bin weight
    bin = Bin(0.0, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
              [vx_min, vy_min, vz_min], [vx_max, vy_max, vz_max], [0.0, 0.0, 0.0],
              1, length(particle_list), 0)
    compute_bin_weight_and_split!(particle_list, bin, split_type)

    # should not merge particles with velocities of different signs
    bin.v_split = [0.0, 0.0, 0.0]
    return bin
end


function octree_merge!(particle_list, bin_list, target_particle_count, max_refinement, aggressiveness, split_type, bin_refinement)
    if (length(particle_list) > target_particle_count)
        do_refinement = true
        target_weight = bin_list[1].w / target_particle_count
        
        while (do_refinement)
            curr_num_particles = length(bin_list)*2
            if (curr_num_particles >= target_particle_count)
                do_refinement = false
            else
                maxweight = 0.0
                refine_bin_id = -1
                    
                if (bin_refinement == Greedy)
                # based on octree paper, attempt to refine bins greedily
                    for (i, bin) in enumerate(bin_list)
                        # check that we haven't reached maximum recursion level
                        if (bin.level < max_refinement)
                            if ((bin.nparticles > 2) && (bin.w > aggressiveness * target_weight))
                                refine_bin_id = i
                                break
                            end
                        end
                    end
                else
                # refine heaviest bin first
                    for (i, bin) in enumerate(bin_list)
                        # check that we haven't reached maximum recursion level
                        if ((bin.level < max_refinement) && (bin.w > maxweight))
                            if ((bin.nparticles > 2) && (bin.w > aggressiveness * maxweight))
                                refine_bin_id = i
                                maxweight = bin.w
                            end
                        end
                    end
                end
                if (refine_bin_id == -1)
                    print("No bins found to refine")
                    do_refinement = false
                else
                    refine_bin!(particle_list, bin_list, refine_bin_id, length(bin_list), split_type)
                end        
            end
        end
    end
        
    for bin in bin_list
        compute_bin_values!(particle_list, bin)
    end
end


function compute_cell!(particle, N_cell, v_lo, v_hi, v_step)
    # for a cell-based merge (assuming N_cell is the same in the x/y/z directions)
    # compute index of cell particle is in
    # if particle is outside the cell region, the index is based on the octant it is in
    if ((particle.v[1] > v_lo[1]) && (particle.v[1] < v_hi[1]) &&
        (particle.v[2] > v_lo[2]) && (particle.v[2] < v_hi[2]) &&
        (particle.v[3] > v_lo[3]) && (particle.v[3] < v_hi[3]))
        
        i_x = (particle.v[1] - v_lo[1]) / v_step[1]
        i_y = (particle.v[2] - v_lo[2]) / v_step[2]
        i_z = (particle.v[3] - v_lo[3]) / v_step[3]
        
        particle.octant = trunc(Int, i_x) * (N_cell^2) + trunc(Int, i_y) * N_cell + trunc(Int, i_z)
        
        particle.octant += 1
    else
        particle.octant = N_cell^3 + 1
        
        if (particle.v[1] > 0.0)
            particle.octant += 1
        end
        if (particle.v[2] > 0.0)
            particle.octant += 2
        end
        if (particle.v[3] > 0.0)
            particle.octant += 4
        end
    end
end


function cell_merge!(particle_list, N_cell, v_lo, v_hi)
    # merge all particles in a cell down to 2 particles
    
    bin_list = Array{Bin}(undef, N_cell^3+8)
    v_step = (v_hi - v_lo) / N_cell
    
    for i in 1: N_cell^3+8
        bin_list[i] = Bin(0.0, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0], 0, 0, 0)
    end
    
    for particle in particle_list
        compute_cell!(particle, N_cell, v_lo, v_hi, v_step)
        
        bin_list[particle.octant].w += particle.w
        
        bin_list[particle.octant].v += particle.w * particle.v
    end
    
    
    for bin in bin_list
        if (bin.w > 0.0)
            bin.v = bin.v / bin.w
        end
    end
    
    for particle in particle_list
        bin_list[particle.octant].c += particle.w * ((particle.v .- bin_list[particle.octant].v).^2)
    end
                            
    
    for bin in bin_list
        if (bin.w > 0.0)
             bin.c = sqrt.(bin.c / bin.w)
        end
    end
                                    
    return bin_list
end


function create_new_particles(bin_list)
    # create new particles from the merging quantities
    new_particles = []
    
    for bin in bin_list
        if (bin.w > 0)
            v_sign = sign.(0.5 .- rand(3))
            
            push!(new_particles, Particle(bin.w / 2, (bin.v .+ (v_sign .* bin.c)), 1))
            push!(new_particles, Particle(bin.w / 2, (bin.v .- (v_sign .* bin.c)), 1))
        end
    end
    
    return new_particles
end


function compute_stats(particle_list, moment_nos)
    # compute stats so that we can check mass, momentum and energy conservation
    ndens = 0.0
    vel = [0.0, 0.0, 0.0]
    
    for particle in particle_list
        ndens += particle.w
        vel += particle.w .* particle.v
    end
    
    vel = vel ./ ndens
    
    print("number density: ", ndens, "\n")
    print("velocity: ", vel, "\n")
    
    moments = fill(0.0, (length(moment_nos), ))
    
    energy = 0.0
     
    for particle in particle_list
        
        energy += particle.w * sum((particle.v .- vel) .^ 2)
    end
    println("Energy: ", energy / ndens)
    
    for particle in particle_list
        
        for (i, moment_no) in enumerate(moment_nos)
            moments[i] += particle.w * sum((particle.v .- vel) .^ moment_no)
        end
    end
    
    for (i, moment_no) in enumerate(moment_nos)
        print("moment no.", moment_no, ": ", moments[i] / ndens, "\n")
    end
end


function create_particles(mass, T, n_particles)
    particle_list = Array{Particle}(undef, n_particles)
    
    for i=1:n_particles
        particle_list[i] = Particle(1.0, maxwellian(T, mass), 0)
    end
    return particle_list
end


function main()
    ar_mass = 6.633521356992e-26;
    T = 1000.0  # temperature of gas
    n_particles = 10000 # number of particles to be sampled

    v_ref = sqrt(2 * constants.k * T / ar_mass)

    moment_nos = [4,6,8,10]


    # octree merging settings
    target_np = 128 # target number of particles
    aggr = 1.0
    max_refinement = 5  # maximum number of times a bin can be split

    inittype = MinMaxVel
    # inittype = LightSpeed

    splittype = MiddleSplit
    # splittype = MeanVelSplit

    refinement = Greedy
    # refinement = Heaviest


    # cell-based merging settings
    # number of cells in each x/y/z velocity direction
    N_cell = 4

    # extent of merging region ([-max_vel_to_v_ref * v_ref, max_vel_to_v_ref * v_ref])
    # scaled w.r.t. v_ref
    max_vel_to_v_ref = 4.5

    particle_list = create_particles(ar_mass, T, n_particles);
    println("Original particles")
    compute_stats(particle_list, moment_nos)
    println()

    println("Octree merge")
    binlist_test = [init_bin(particle_list, inittype, splittype)];
    @time octree_merge!(particle_list, binlist_test, target_np, 5, aggr, splittype, refinement);
    newparticles = create_new_particles(binlist_test);

    compute_stats(newparticles, moment_nos)
    println()

    println("Cell-based merge")
    @time binlist_test = cell_merge!(particle_list, N_cell,
                                     [-v_ref * max_vel_to_v_ref, -v_ref * max_vel_to_v_ref, -v_ref * max_vel_to_v_ref],
                                     [v_ref * max_vel_to_v_ref, v_ref * max_vel_to_v_ref, v_ref * max_vel_to_v_ref]);
    newparticles = create_new_particles(binlist_test);

    compute_stats(newparticles, moment_nos)
end

main()