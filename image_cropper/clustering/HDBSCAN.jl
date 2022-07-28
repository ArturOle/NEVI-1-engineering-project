

mutable struct Point
    label::Int64
    coordinates::Vector{Float64}
end

function prim(graph::Graph)
    msp = zeros(Int64, (length(graph.points)-1,3))
    visited = msp[:, 2]
   
    for i in 1:length(graph.points)-1
        weight = typemax(Int64)
        best_goto = 0
        best_from = 0
        for point_id in visited
            if point_id != 0
                point = graph.points[point_id]
                for path in point.connections
                    if path.connection.first == point.id && path.connection.second ∉ visited
                        if path.weight < weight
                            weight = path.weight
                            best_goto = path.connection.second
                            best_from = point_id
                        end
                    elseif path.connection.first ∉ visited
                        if path.weight < weight
                            weight = path.weight
                            best_goto = path.connection.first
                            best_from = point_id
                        end
                    end
                end
            end
        end
        msp[i, 1] = best_from
        msp[i, 2] = best_goto
        msp[i, 3] = weight

    end

    return msp
end


function HDBSCAN()

end
