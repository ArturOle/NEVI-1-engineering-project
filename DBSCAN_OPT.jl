using Random
using Distributions
using Plots

mutable struct Point
    label::Int64
    coordinates::Vector{Float64}
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
                    if seed.label < 1
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

    for i in 1:length(p1)
        sum += abs2(p1[i] - p2[i])
    end

    return sqrt(sum)
end

function euclidian_distances(v1, v2)
    v1 = [v1[i, :] for i in 1:size(v1,1)]
    v2 = [v2[i, :] for i in 1:size(v2,1)]
    s = size(v1,1)
    d = zeros((length(v1[1]),length(v2[2])))
    
    for j in 1:length(v2[2])
        for k in 1:length(v1[1])
            summed = 0
            for i in 1:s
                summed += (v1[i][k] - v2[i][j])^2
            end
            d[k, j] = sqrt(summed)
        end
    end

    return d'
end


data = zeros(Float64, 3, 15000)
for i in 1:5000
    data[1, i] = rand(Normal(100, 10))
    data[2, i] = rand(Normal(40, 5))
    data[3, i] = rand(Normal(40, 10))
end

for i in 5000:10000
    data[1, i] = rand(Normal(20, 4))
    data[2, i] = rand(Normal(50, 6))
    data[3, i] = rand(Normal(50, 10))
end

for i in 10000:15000
    data[1, i] = rand(Normal(50, 5))
    data[2, i] = rand(Normal(90, 10))
    data[3, i] = rand(Normal(70, 8))
end

# display(data)
plotly()
scatter()
@time ( clustered, cluster_counter ) = DBSCAN(data, 4, 6)
display( clustered )
for i in -1:cluster_counter 
    cluster = [(x.coordinates[1], x.coordinates[2], x.coordinates[3]) for x in clustered if x.label == i]
    scatter!(cluster, markersize=1.5)
end
display(current())

# @time find_neighbors(data, data[:, 1], 10)
