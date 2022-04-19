# using Distributions
using Plots
using Statistics
# using Profile

using Colors
using Images
using FileIO

""" NOTE:
Current implementation wrongly crop the images with black corners.
Possible solutions:
    - Different distance calculating function for choosing 
      the closest cluster to the center. ( maybe decrease the relevancy of pixels futher from the center )

"""

const MINIMUM_SIZE = (128, 128)

function img_to_graph(image_name::String="HAM10000_images\\ISIC_0024944.jpg")
    """
    Function transforming image into the matrix of points  
    """
    
    image = load("datasets\\HAM10000\\$image_name")
    cv = channelview(image)
    s = size(cv)
    max_x = s[3]
    max_y = s[2]

    img = zeros(Float64, 5, max_x*max_y)

    for j in 1:max_y
        for i in 1:max_x
            img[1, (j-1)*max_x + i] = cv[1, j, i]
            img[2, (j-1)*max_x + i] = cv[2, j, i]
            img[3, (j-1)*max_x + i] = cv[3, j, i]
            img[4, (j-1)*max_x + i] = i
            img[5, (j-1)*max_x + i] = j

        end
    end

    return img
end

function show_partition_matrix(partition_matrix)
    """
    Pretty print for partition matrix
    """
    for (i,y) in enumerate(eachrow(partition_matrix))
        println(y, " Ω", i)
    end
    print(join([join((" x", i)) for i in 1:length(eachcol(partition_matrix))]), "\n")
end

function fuzzy_c_means(data, number_of_clusters, m, quiet::Bool, ϵ=1e-3)
    """
    Quiet implementation of Fuzzy c-means with random inicialization and Gaussian distances
    prepared to work with multi-dimentional data.
    """
    class_data = data[1:3, :]
    U = zeros(Float16, number_of_clusters, size(data,2))
    
    U = init_partition_matrix(U)
    V = cluster_centers(U, class_data, m)
    
    distances = euclidian_distances(V, class_data)
    U_prev = U

    U = update_partition_table(U, distances, m)

    while abs(frobenius_norm(U_prev, U)) > ϵ
        V = cluster_centers(U, class_data, m)
        distances = euclidian_distances(V, class_data)
        U_prev = U
        U = update_partition_table(U, distances, m)
    end

    classified = classified_pixel(U, data)

    return (classified, U, V)
    
end

function fuzzy_c_means(data, number_of_clusters, m, ϵ=1e-3)
    """
    Implementation of Fuzzy c-means with random inicialization and Gaussian distances
    prepared to work with multi-dimentional data with visualization.
    """
    class_data = data[1:3, :]
    
    U = zeros(Float16, number_of_clusters, size(data,2))
    
    U = init_partition_matrix(U)
    V = cluster_centers(U, class_data, m)
    
    distances = euclidian_distances(V, class_data)
    U_prev = U

    U = update_partition_table(U, distances, m)

    #J = criterion_function(U, distances)

    while abs(frobenius_norm(U_prev, U)) > ϵ
        V = cluster_centers(U, class_data, m)
        distances = euclidian_distances(V, class_data)
        U_prev = U
        U = update_partition_table(U, distances, m)
        #J = criterion_function(U, distances)
    end

    classified = classified_pixel(U, data)
    # display(classified)

    scatter(legend=false)

    for i in 1:number_of_clusters
        scatter_array_1 = []
        scatter_array_2 = []
        scatter_array_3 = []
        for j in 1:length(data[1, :])
            if classified[j][2] == i
                append!(scatter_array_1, classified[j][1][4])
                append!(scatter_array_2, classified[j][1][5])
                append!(scatter_array_3, classified[j][1][3])
            end
        end
        scatter!(scatter_array_1, scatter_array_2, alpha=1., msize=1, markerstrokewidth=0)
        
    end

    # scatter!(scatter_array_1, scatter_array_2, scatter_array_3)
    # scatter!(V[2,:], V[3, :])
    display(current())
    return (classified, U, V)
    
end

function classified_pixel(U, data)
    classified = []

    for (i, x) in enumerate(eachcol(U))
        append!(classified, [[data[:,i], argmax(x)]])
    end

    return classified
end

function init_partition_matrix(partition_table)
    height = length(partition_table[:, 1])
    len = length(partition_table[1, :])

    for col in 1:len
        for row in 1:rand(1:height)
            partition_table[rand(1:height), col] += 1
        end
    end

    for row in 1:height
        if sum(partition_table[row, :]) == 0
            partition_table[row, rand(1:height)] += 1
        end
    end

    for i in 1:len
        col_sum = sum(partition_table[:, i])
        partition_table[:, i] = partition_table[:, i] ./ col_sum
    end

    return partition_table
end

function init_random_partition_matrix(partition_matrix)
    for (k, member) in enumerate(eachcol(partition_matrix))
        resources = 1
        for i in 1:length(member)-1
            value = rand(Uniform(0, resources))
            resources -= value
            partition_matrix[i, k] = value
        end
        partition_matrix[end, k] = resources
    end

    return partition_matrix
end

function cluster_centers(partition_table, data)  #  prototype
    """
    Calculating centroid center prototypes for hard clustering
    """
    size_of_data = size(data)
    v = zeros(size(partition_table, 1), size_of_data[1])   

    for partition_table_row in 1:size(partition_table, 1)
        for data_row in 1:size_of_data[1]
            summed_nominator = 0
            summed_denominator = 0
            for entry_number in 1:size_of_data[2]
                summed_denominator += partition_table[partition_table_row, entry_number]
                summed_nominator += data[data_row, entry_number]*partition_table[partition_table_row, entry_number]
            end
            v[partition_table_row, data_row] = summed_nominator/summed_denominator
        end     
    end

    return rotr90(v)
end

function cluster_centers(partition_table, data, m) 
    """
    Calculating centroid center prototypes for fuzzy clustering
    """
    size_of_data = size(data)
    v = zeros(size(partition_table, 1), size_of_data[1])   

    for partition_table_row in 1:size(partition_table, 1)
        for data_row in 1:size_of_data[1]
            summed_nominator = 0
            summed_denominator = 0
            for entry_number in 1:size_of_data[2]
                fuzzied_pt = (partition_table[partition_table_row, entry_number]^m)
                summed_denominator += fuzzied_pt
                summed_nominator += data[data_row, entry_number]*fuzzied_pt
            end
            v[partition_table_row, data_row] = summed_nominator/summed_denominator
        end     
    end
    return rotr90(v)
end

function frobenius_norm(m1, m2)
    summed_first = 0
    summed_second = 0
    for i in 1:size(m1,2)
        for j in 1:size(m2,1)
            summed_first += m1[j,i]^2
            summed_second += m2[j,i]^2
        end
    end
    return sqrt(summed_first) - sqrt(summed_second)
end

function euclidian_distances(v1, v2)
    """
    Calculating distances between two points in multiple dimentions which
    are described as vectors
    """
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
            d[ k, j] = sqrt(summed)
        end
    end

    return d'
end

function euclidian_distances_sqrd(v1::Matrix, v2::Matrix)
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
            d[ k, j] = summed
        end
    end

    return d'
end

function update_partition_table(partition_table, distances, m)
    partition_table = zeros(Float64, size(partition_table,1), size(partition_table,2))
    height = length(partition_table[:, 1])
    len = length(partition_table[1, :])

    for col in 1:len
        # Time critical! Need for optimiziation
        dist_sum = sum([x^(2/(1-m)) for x in distances[col, :]])
        if 1 in partition_table[:, col]   
            set = findall(x->x==1, partition_table[:, col]) 
            for row in 1:height
                if row in set
                    partition_table[row, col] = 1
                else
                    partition_table[row, col] = 0
                end
            end
        else
            # time critical
            for row in 1:height
                partition_table[row, col] = (distances[col, row]^(2/(1-m)))/dist_sum
            end
        end

    end

    return partition_table
end

function criterion_function(partition_table, distances)
    J = 0
    for column in 1:size(partition_table,2)
        for row in 1:size(partition_table,1)
            J += partition_table[row, column]*distances[row, column]
        end
    end
    return J
end

function init_random(partition_table)
    height = length(eachrow(partition_table))
    for (i, _) in enumerate(eachcol(partition_table))
        partition_table[rand(1:height), i] = 1
    end

    for (i, row) in enumerate(eachrow(partition_table))
        if sum(row) == 0
            rand_col = rand(1:size(partition_table, 2))
            partition_table[:, rand_col] .= 0
            partition_table[i, rand_col] = 1
        end
        i = 1
    end
    return partition_table
end

function middle(img_size::Tuple)
    img_size_h = img_size[1]
    img_size_v = img_size[2]

    middle_h = Int64(floor(img_size_h/2))
    middle_v = Int64(floor(img_size_v/2))

    return (middle_h, middle_v)
end

function middle_cluster(number_of_clusters, img_size, data)
    mid = middle(img_size)
    decision_dictionary = Dict{Int, Vector{Float64}}([[x,[]] for x in 1:number_of_clusters])

    for point in data[1]
        append!(decision_dictionary[point[end]], [sqrt((mid[1]-point[1][end-1])^2 + (mid[2] - point[1][end])^2)])
    end

    [decision_dictionary[k] = [mean(x)] for (k,x) in decision_dictionary]
    return findmin(decision_dictionary)
end

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

function density_clustering(db)
    (clustered, cluster_counter) = DBSCAN(db, 7, 5)
    scatter()

    main_cluster = Pair(0, 0)
    for i in -1:cluster_counter 
        cluster = [(x.coordinates[1], x.coordinates[2]) for x in clustered if x.label == i]
        
        if length(cluster) > length(main_cluster[2])
            main_cluster = (i=>cluster)
        end

        scatter!(cluster, label="cluster_label: $i", markersize=3, markerstrokewidth=0)
    end
    display(current())

    return main_cluster
    #return (findminmax(cluster_vector_x), findminmax(cluster_vector_y))
end

function density_clustering(db, quiet::Bool)
    (clustered, cluster_counter) = DBSCAN(db, 7, 5)

    main_cluster = Pair(0, 0)
    for i in -1:cluster_counter 
        cluster = [(x.coordinates[1], x.coordinates[2]) for x in clustered if x.label == i]

        if length(cluster) > length(main_cluster[2])
            main_cluster = (i=>cluster)
        end
    end

    return main_cluster
end

function findminmax(arr::Vector)
    max = -Inf
    min = Inf
    if length(arr)%2 == 0
        for i in 1:2:length(arr)
            if arr[i] > arr[i+1]
                if arr[i] > max
                    max = arr[i]
                end
                if arr[i+1] < min
                    min = arr[i+1]
                end
            else 
                if arr[i+1] > max
                    max = arr[i+1]
                end
                if arr[i] < min
                    min = arr[i]
                end
            end
        end
    else
        for i in 1:2:length(arr)-1
            if arr[i] > arr[i+1]
                if arr[i] > max
                    max = arr[i]
                end
                if arr[i+1] < min
                    min = arr[i+1]
                end
            else 
                if arr[i+1] > max
                    max = arr[i+1]
                end
                if arr[i] < min
                    min = arr[i]
                end
            end
        end
    end
    return (Int(min), Int(max))
end

function crop(img, initial_range, img_size, minimum_size=(256, 256), border=(20, 20))
    projected_size = (initial_range[1][2]-initial_range[1][1], initial_range[2][2]-initial_range[2][1])
    display(projected_size)

    if projected_size < minimum_size
        println("$projected_size is smaller than $MINIMUM_SIZE")
        projected_size = MINIMUM_SIZE
        calculated_range = (
            (Int(ceil(initial_range[1][1] - MINIMUM_SIZE[2]/5)), Int(floor(initial_range[1][2] + MINIMUM_SIZE[2]/5))),
            (Int(ceil(initial_range[2][1] - MINIMUM_SIZE[1]/5)), Int(floor(initial_range[2][2] + MINIMUM_SIZE[1]/5)))
        )
    else
        println("$projected_size is greated than $MINIMUM_SIZE")

        # ERROR - For reimplementation
        if initial_range[1][1]-border[1]*2 > 0 && initial_range[2][2]+border[1]*2 < img_size[2] && initial_range[2][1]-border[1]*2 > 0 && initial_range[1][2]+border[1]*2 < img_size[1]
            calculated_range = (
                (initial_range[1][1]-border[1], initial_range[1][2]+border[1]),
                (initial_range[2][1]-border[2], initial_range[2][2]+border[2])
            )
        else
            calculated_range = (
                (1, img_size[1]),
                (1, img_size[2])
            )
        end
    end

    img = @view img[calculated_range[2][1]:calculated_range[2][2], calculated_range[1][1]:calculated_range[1][2]]
    display(img)

end

function extract_dimentions(choosen_cluster::Int64, data)
    cluster_vector_x = Vector{Int}()
    cluster_vector_y = Vector{Int}()

    for point in data[1]
        if point[end] == choosen_cluster
            append!(cluster_vector_x, [point[end-1][end-1]])
            append!(cluster_vector_y, [point[end-1][end  ]])
        end
    end
    return hcat(cluster_vector_x, cluster_vector_y)'
end

function processing(image_name::String="HAM10000_images\\ISIC_0024943.jpg")
    image = load("datasets\\HAM10000\\$image_name")
    image = imresize(image, ratio=2/5)
    cv = channelview(image)
    s = size(cv)
    max_x = s[3]
    max_y = s[2]

    img = zeros(Float64, 5, max_x*max_y)

    for j in 1:max_y
        for i in 1:max_x
            img[1, (j-1)*max_x + i] = cv[1, j, i]
            img[2, (j-1)*max_x + i] = cv[2, j, i]
            img[3, (j-1)*max_x + i] = cv[3, j, i]
            img[4, (j-1)*max_x + i] = i
            img[5, (j-1)*max_x + i] = j
        end
    end     
        
    nr_of_clusters = 5
    img_size = (max_x, max_y)
        
    data = fuzzy_c_means(img, nr_of_clusters, 1.3)


    # # # # # # # # # # # #
    #  HDBSCAN/FCMED here #
    # # # # # # # # # # # #
    # distance clustering with anomalies detection 
    # after color clustering
    

    choosen_cluster = middle_cluster(nr_of_clusters, img_size, data)[2]
    db = extract_dimentions(choosen_cluster, data)
    main_cluster = density_clustering(db)

    x = [i[2] for i in main_cluster[2]]
    y = [i[1] for i in main_cluster[2]]
    cluster_boundries = (findminmax(y), findminmax(x))
    display(cluster_boundries)
    # cluster_boundries(choosen_cluster, data)
    cropped_image = crop(image, cluster_boundries, img_size, MINIMUM_SIZE, (5, 1))
    display(cropped_image)
end


function processing(quiet::Bool, image_name::String="HAM10000_images\\ISIC_0024943.jpg")
    image = load("datasets\\HAM10000\\$image_name")
    image = imresize(image, ratio=2/5)
    cv = channelview(image)
    s = size(cv)
    max_x = s[3]
    max_y = s[2]

    img = zeros(Float64, 5, max_x*max_y)

    for j in 1:max_y
        for i in 1:max_x
            img[1, (j-1)*max_x + i] = cv[1, j, i]
            img[2, (j-1)*max_x + i] = cv[2, j, i]
            img[3, (j-1)*max_x + i] = cv[3, j, i]
            img[4, (j-1)*max_x + i] = i
            img[5, (j-1)*max_x + i] = j
        end
    end     
        
    nr_of_clusters = 5
    img_size = (max_x, max_y)
        
    data = fuzzy_c_means(img, nr_of_clusters, 1.3, true)


    # # # # # # # # # # # #
    #  HDBSCAN/FCMED here #
    # # # # # # # # # # # #
    # distance clustering with anomalies detection 
    # after color clustering
    

    choosen_cluster = middle_cluster(nr_of_clusters, img_size, data)[2]
    db = extract_dimentions(choosen_cluster, data)
    main_cluster = density_clustering(db, true)

    x = [i[2] for i in main_cluster[2]]
    y = [i[1] for i in main_cluster[2]]
    cluster_boundries = (findminmax(y), findminmax(x))
    display(cluster_boundries)
    # cluster_boundries(choosen_cluster, data)
    cropped_image = crop(image, cluster_boundries, img_size, MINIMUM_SIZE, (5, 1))
    display(cropped_image)
end

# processing()
