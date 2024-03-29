using Distributions
using Plots
using BenchmarkTools


mutable struct Point
    label::Int64
    coordinates::Vector{Float64}
end


function middle(range)

    range_h = range[1]
    range_v = range[2]

    middle_h = Int64(floor(range_h/2))
    middle_v = Int64(floor(range_v/2))

    return (middle_h, middle_v)
end


function find_neighbors(data, point, eps)

    number_of_neighbors = 0
    neighbors = []

    for second_point in data
        if point != second_point
            if euclidian_distance_alt(point.coordinates, second_point.coordinates) < eps
                number_of_neighbors += 1
                append!(neighbors, [second_point])
            end
        end
    end

    return (number_of_neighbors, neighbors)
end


function euclidian_distance_alt(p1, p2)
    sum = 0

    for i in eachindex(p1)
        sum += abs2(p1[i] - p2[i])
    end

    return sqrt(sum)
end


function DBSCAN(data, eps, min_cluster_size)

    cluster_counter = 0
    point_data = [Point(0, p) for p in eachcol(data)]

    for point in point_data
        if point.label == 0
            (neighbor_counter, neighbors) = find_neighbors(point_data, point, eps)
            if neighbor_counter < min_cluster_size
                point.label = -1
            else
                cluster_counter += 1
                point.label = cluster_counter
                for seed in neighbors
                    if seed.label == 0
                        seed.label = cluster_counter
                    elseif seed.label == -1
                        seed.label = cluster_counter
                    end
                    (neighbor_counter, seed_neighbors) = find_neighbors(point_data, seed, eps)
                    if neighbor_counter >= min_cluster_size
                        union!(neighbors, seed_neighbors)
                    end
                end
            end
        end
    end

    return (point_data, cluster_counter)
end


function density_clustering(db, img_size; eps=6.2, cluster_size=30)

    (clustered, cluster_counter) = DBSCAN(db, eps, cluster_size)
    mid = middle(img_size)
    decision_dictionary = Dict{Int, Vector{Float64}}([[x,[]] for x in 1:cluster_counter])

    scatter(legend=true)
    for i in -1:cluster_counter 
        cluster = [(x.coordinates[1], x.coordinates[2]) for x in clustered if x.label == i]
        scatter!(cluster, label="clu: $i", markersize=6, markerstrokewidth=0)
    end
    yflip!(true)
    display(current())

    for point in clustered
        if point.label > 0
            append!(
                decision_dictionary[point.label], 
                [sqrt(sum([(mid[i]-point.coordinates[i])^2 for i in 1:length(point.coordinates)]))]
            )
        end
    end

    for (k,x) in decision_dictionary
        decision_dictionary[k] = [mean(x)]
    end

    best = findmin(decision_dictionary)
    return [best[2],[(x.coordinates[1], x.coordinates[2]) for x in clustered if x.label == best[2]]]
end

function density_clustering(db, img_size, quiet::Bool; eps=6.2, cluster_size=30)

    (clustered, cluster_counter) = DBSCAN(db, eps, cluster_size)
    mid = middle(img_size)
    decision_dictionary = Dict{Int, Vector{Float64}}([[x,[]] for x in 1:cluster_counter])

    for point in clustered
        if point.label > 0
            append!(
                decision_dictionary[point.label], 
                [sqrt(sum([(mid[i]-point.coordinates[i])^2 for i in 1:length(point.coordinates)]))]
            )
        end
    end

    for (k,x) in decision_dictionary
        decision_dictionary[k] = [mean(x)]
    end

    best = findmin(decision_dictionary)
    result = filtered_best_coordinates(clustered, best)
    return result
end

function filtered_best_coordinates(clustered, best)
    best_coordinates = [
        (x.coordinates[1], x.coordinates[2]) for x in clustered if x.label == best[2]
    ]
    return [best[2], best_coordinates]
end
