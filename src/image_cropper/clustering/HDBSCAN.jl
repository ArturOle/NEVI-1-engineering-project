using Distributions
using BenchmarkTools
include("utils/graph.jl")

# DISCLAIMER: Work In Progress

function find_elem(array, elem)
    for (i, elin) in enumerate(array) 
        if elin == elem
            return i   
        end
    end
    return 0
end

function adjacency_matrix(data::Matrix)
    
    adj_size = size(data, 1)
    adjacency_matrix = zeros(Float64, (adj_size, adj_size))

    for (i, row_i) in enumerate(eachrow(data))
        for (j, row_j) in enumerate(eachrow(data))
            if i != j
                if adjacency_matrix[i,j] == 0
                    dist = euclidian_distance_alt(row_i, row_j)
                    adjacency_matrix[i,j] = dist
                    adjacency_matrix[j,i] = dist
                end
            end
        end
    end
    
    return adjacency_matrix
end

function adjacency_matrix_alt(data::Matrix)
    adj_size = size(data, 1)
    adjacency_matrix = zeros(Float64, (adj_size, adj_size))

    all_rows = enumerate(eachrow(data))
    for (i, row_i) in all_rows
        for (j, row_j) in all_rows
            if i != j
                if adjacency_matrix[i,j] == 0
                    dist = euclidian_distance_alt(row_i, row_j)
                    adjacency_matrix[i,j] = dist
                    adjacency_matrix[j,i] = dist
                end
            end
        end
    end

    return adjacency_matrix
end

function prim(data::Matrix)
    adm = adjacency_matrix(data)
    display(adm)
    # while edge_number < size(data,2)
    #     edge_number = 1
    #     data[edge_number, 3] = 1 
    # end
    # for first_point in eachrow(data)

    #     min_value = Inf
    #     x = 0
    #     y = 0
    #     for second_point in eachrow(data)

    #     end
    # end
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

    # display(msp)
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
    # display(graph.points[1:5])
    minimum_spanning_tree = prim(graph)
end

function test_env()
    data = setup()
    display(adjacency_matrix_alt(data))
    @benchmark adjacency_matrix_alt(data)
    # @benchmark adjacency_matrix_alt(data) setup=(data=setup())
    # @time HDBSCAN(generate_map(data), 4)
end

function setup()
    data = Matrix{Int64}(undef, 3000, 2)

    for i=1:100
        data[i, 1] = Int(floor(rand(Normal(1 ,40))))
        data[i, 2] = Int(floor(rand(Normal(1 ,40))))

    end
    for i=100:200
        data[i, 1] = Int(floor(rand(Normal(150 ,10))))
        data[i, 2] = Int(floor(rand(Normal(15 ,10))))

    end
    for i=200:300
        data[i, 1] = Int(floor(rand(Normal(300 ,15))))
        data[i, 2] = Int(floor(rand(Normal(300, 5))))

    end
    return data
end

test_env()
