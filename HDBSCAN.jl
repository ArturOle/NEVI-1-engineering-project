using Random
using Distributions
using Plots


function HDBSCAN(data, min_cluster_size)

end

function DBSCAN(data, eps, min_cluster_size)
    cluster_counter = 0
    point_labels = Dict()

    for point in eachcol(data)
        if point ∉ keys(point_labels)
            (neighbor_counter, neighbors) = find_neighbors(data, point, eps)
            if neighbor_counter < min_cluster_size
                point_labels[point] = -1
            else
                cluster_counter += 1
                point_labels[point] = cluster_counter
                for seed in neighbors
                    if seed ∉ keys(point_labels)
                        point_labels[seed] = cluster_counter
                    else
                        point_labels[seed] = cluster_counter
                    end
                    (neighbor_counter, seed_neighbors) = find_neighbors(data, seed, eps)
                    if neighbor_counter >= min_cluster_size
                        union!(neighbors, seed_neighbors)
                    end
                end
            end
        end
    end
    return point_labels
end

# Function is saving the network of connections between neighbors
function find_neighbors(data, point, eps)
    number_of_neighbors = 0
    neighbors = []

    for second_point in eachcol(data)
        if point != second_point
            if euclidian_distances(point, second_point)[1] < eps
                number_of_neighbors += 1
                append!(neighbors, [second_point])
            end
        end
    end

    return (number_of_neighbors, neighbors)
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



data = zeros(Float64, 3, 600)
for i in 1:200
    data[1, i] = rand(Normal(100, 10))
    data[2, i] = rand(Normal(40, 5))
    data[3, i] = rand(Normal(40, 10))
end

for i in 200:400
    data[1, i] = rand(Normal(20, 4))
    data[2, i] = rand(Normal(50, 6))
    data[3, i] = rand(Normal(50, 10))
end

for i in 400:600
    data[1, i] = rand(Normal(50, 5))
    data[2, i] = rand(Normal(90, 10))
    data[3, i] = rand(Normal(70, 8))
end

# display(data)
# scatter()
@time clustered = DBSCAN(data, 9, 5)
# for i in -1:maximum(values(clustered))
#     cluster = [(x[1], x[2], x[3]) for x in keys(clustered) if clustered[x] == i]
#     scatter!(cluster, markersize=1.5)
# end
# display(current())

# @time find_neighbors(data, data[:, 1], 10)

