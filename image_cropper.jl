using Distributions
# using Random
# using CSV
# using DataFrames
using Plots
using Profile

using StaticArrays
using Colors
using Images
using FileIO



function img_to_graph(image_name::String="ISIC_0024944.jpg")
    image = load("img_for_inż\\$image_name")
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

    display(img)
    return img
end


function show_partition_matrix(partition_matrix)
    for (i,y) in enumerate(eachrow(partition_matrix))
        println(y, " Ω", i)
    end
    print(join([join((" x", i)) for i in 1:length(eachcol(partition_matrix))]), "\n")
end

function fuzzy_c_means(data, number_of_clusters, m)
    println("clustering start")
    class_data = data[1:3, :]
    ϵ = 1e-2
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
    println("clustering done")

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
        scatter!(scatter_array_1, scatter_array_2, alpha=0.3, msize=1)
        
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
    # time critical
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
    display(findmin(decision_dictionary))
end

nr_of_clusters = 5
img_size = (600, 450)
fuzzy_c_means(img_to_graph("ISIC_0024940.jpg"), nr_of_clusters, 1.3)

# @profile fuzzy_c_means(img_to_graph("ISIC_0024940.jpg"), nr_of_clusters, 1.3)

# Profile.print(format=:flat)
# display(data[1])  # Classes and the pixels included

# middle_cluster(nr_of_clusters, img_size, data)
