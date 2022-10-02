include("utils/graph.jl")
using Distributions


function find_elem(array, elem)
    for (i, elin) in enumerate(array) 
        if elin == elem
            return i   
        end
    end
    return 0
end

function prim(graph::Graph)
    bad_path = UndirectedPath(-1, maxintfloat(Float64), Pair(-1,-1))
    unvisited_points = copy(graph.points)
    visited_points = [popfirst!(unvisited_points)]
    msp = Vector{UndirectedPath}()
   
    while length(unvisited_points) >= 1
        second_point_id = -1
        shortest_path = bad_path
        for point in visited_points
            for path in point.connections
                if path.connection.first == point.id
                    if graph.points[path.connection.second] in unvisited_points
                        if path.weight < shortest_path.weight
                            shortest_path = path
                            second_point_id = path.connection.second
                        end
                    end
                else
                    if graph.points[path.connection.first] in unvisited_points
                        if path.weight < shortest_path.weight
                            shortest_path = path
                            second_point_id = path.connection.first
                        end
                    end
                end
            end
        end
        append!(msp, [shortest_path])
        index = find_elem([point.id for point in unvisited_points], second_point_id)
        append!(visited_points, [unvisited_points[index]])
        deleteat!(unvisited_points, index)
    end

    display(msp)
end

function euclidian_distance_alt(p1, p2)
    sum = 0

    for i in eachindex(p1)
        sum += abs2(p1[i] - p2[i])
    end

    return sqrt(sum)
end

function distance_matrix(data::Matrix)
    data_size_x = size(data)[1]
    distances = zeros(Float64, data_size_x, data_size_x)
    distances[1:end, 1:end] .= Inf64
    for i in 1:data_size_x
        for j in 1:data_size_x
            distances[i, j] = euclidian_distance_alt(data[i, :], data[j, :])
        end
    end

    return distances
end

function HDBSCAN(data::Matrix)
    mutual_reachability_graph = generate_map(data)
    minimum_spanning_tree = prim(graph)
end

function HDBSCAN(graph::Graph, m_pts::Int64) # m_pts - minimum number of points
    
    for point in graph.points
        sort!(point.connections, by=up->up.weight)
        point.core_distance = sum(up->up.weight, point.connections[1:m_pts])
    end
    display(graph.points[1:5])
    minimum_spanning_tree = prim(graph)
end

function test_env()

    data = Matrix{Int64}(undef, 30, 2)

    for i=1:10
        data[i, 1] = Int(floor(rand(Normal(1 ,40))))
        data[i, 2] = Int(floor(rand(Normal(1 ,40))))

    end
    for i=10:20
        data[i, 1] = Int(floor(rand(Normal(150 ,10))))
        data[i, 2] = Int(floor(rand(Normal(15 ,10))))

    end
    for i=20:30
        data[i, 1] = Int(floor(rand(Normal(300 ,15))))
        data[i, 2] = Int(floor(rand(Normal(300, 5))))

    end

    @time HDBSCAN(generate_map(data), 4)
end

test_env()