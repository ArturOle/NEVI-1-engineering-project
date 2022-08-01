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

    return msp
end


function HDBSCAN()

end


data = Matrix{Int64}(undef, 300, 2)


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

@time prim(generate_map(data))
