using Combinatorics


mutable struct UndirectedPath
    id::Int
    weight::Real
    connection::Pair{Int,Int}
end

mutable struct Point
    id::Int
    value::Float16
    core_distance::Float64
    coordinates::Vector{Real}
    connections::Vector{UndirectedPath}
end

Point(id, value) = Point(id, value, 0, [], [])
Point(id, value, coordinates) = Point(id, value, 0, coordinates, [])

mutable struct Graph
    points::Vector{Point}
    paths::Vector{UndirectedPath}
end

function find_path(a::Point, b::Point)
    for path in a.connections
        if  path.connection.first == b.id ||
            path.connection.second == b.id
            return path
        end
    end
    for path in b.connections
        if  path.connection.first == a.id ||
            path.connection.second == a.id
            return path
        end
    end
end


function point_at(graph::Graph, point_id::Int)
    for point in graph.points
        if point.id == point_id
            return point
        end
    end
    println("No point witch such ID.")
    return NaN
end

function path_at(graph::Graph, path_id::Int)
    for path in graph.paths
        if path.id == path_id
            return path
        end
    end
    println("No path witch such ID.")
    return NaN
end

function find_all_connections(graph::Graph, point::Point)
    for path in graph.paths
        if path.connection.first == point.id || path.connection.second == point.id
            append!(point.connections, [path])
        end
    end
end

function find_all_paths_with_point(graph::Graph, point::Point)
    conns = [] 
    for path in graph.paths
        if path.connection.first == point.id || path.connection.second == point.id
            append!(conns, [path])
        end
    end
    return conns
end

function find_all_connections(paths::Vector{UndirectedPath}, point::Point)
    for path in paths
        if path.connection.first == point.id || path.connection.second == point.id
            append!(point.connections, [path])
        end
    end
end

function find_all_paths_with_point(paths::Vector{UndirectedPath}, point::Point)
    conns = [] 
    for path in paths
        if path.connection.first == point.id || path.connection.second == point.id
            append!(conns, [path])
        end
    end
    return conns
end

function point_id(graph::Graph, point_at::Point)
    for index in 1:length(graph.points)
        if graph.points[index] == point_at
            return graph.points[index].id
        end
    end
end

function test_graph(graph::Graph)
    println(graph.points)
    println(graph.paths)
end


"""
Generator of graphs based on given points coordinates
"""
function generate_map(x_coordinates, y_coordinates)
	
	points = Vector{Point}()
	paths = Vector{UndirectedPath}()
	len_x = length(x_coordinates)
	
	if len_x == length(y_coordinates)
		for i in 1:len_x
			append!(points, [Point(i, 0, [x_coordinates[i], y_coordinates[i]])])
		end

		path_id = 1
		for i in 1:len_x
			for j in 1:len_x
				if points[i] != points[j]
					create = true
					for path in paths
						if  (path.connection.first == points[i].id || 
							path.connection.second == points[i].id) &&
							(path.connection.first == points[j].id || 
							path.connection.second == points[j].id)
							
							create = false
							break
						end
					end
					if create == true
						path = UndirectedPath(
                            path_id,
                            Pair(points[i].id, points[j].id),
                            sqrt(abs(x_coordinates[i]-x_coordinates[j])^2 + abs(y_coordinates[i]-y_coordinates[j])^2)
                        )
						append!(paths, [path])
						path_id +=1
					end
				end
			end
		end
		for point in points
			for path in paths
				if path.connection.first == point.id  || path.connection.second == point.id
					append!(point.connections, [path])
				end
			end
		end

	end

	return Graph(points, paths)
end

"""
[only 2d] Generator of graphs based on given points coordinates
"""
function generate_map(coordinates)
	len_x = length(coordinates[:, 2])
	points = Vector{Point}(undef, len_x)
	number_of_paths = 0
	for x in 1:len_x-1
		number_of_paths += x
	end
	paths = Vector{UndirectedPath}(undef, number_of_paths)
	
	
	if len_x == length(coordinates[:, 1])
		for i in 1:len_x
			points[i] = Point(i, 0, [coordinates[i, 2], coordinates[i, 1]])
		end

		path_id = 1
		for i in 1:len_x
			for j in 1:i
				if points[i] != points[j]
					path = UndirectedPath(
                        path_id,
                        sqrt(abs(coordinates[i, 1]-coordinates[j, 1])^2 + abs(coordinates[i, 2]-coordinates[j, 2])^2),
                        Pair(points[i].id, points[j].id)
                    )
					paths[path_id] = path
					path_id +=1
				end
			end
		end

		for point in points
			for path in paths
				if point.id in path.connection
					append!(point.connections, [path])
				end
			end
		end

	end
	
	return Graph(points, paths)
end
