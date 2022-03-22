using Distributions
using Random
using CSV
using DataFrames
using Plots


function show_partition_matrix(partition_matrix)
    for (i,y) in enumerate(eachrow(partition_matrix))
        println(y, " Ω", i)
    end
    print(join([join((" x", i)) for i in 1:length(eachcol(partition_matrix))]), "\n")
end

function fuzzy_c_means(data, number_of_clusters, m)
    ϵ = 1e-26
    U = zeros(Float64, number_of_clusters, size(data,2))
    
    U = init_partition_matrix(U)
    V = cluster_centers(U, data, m)
    
    distances = euclidian_distances(V, data)
    U_prev = U
    U = update_partition_table(U, distances, m)

    J = criterion_function(U, distances)

    while abs(frobenius_norm(U_prev, U)) > ϵ
        V = cluster_centers(U, data, m)
        distances = euclidian_distances(V, data)
        U_prev = U
        U = update_partition_table(U, distances, m)
        J = criterion_function(U, distances)
    end
    display(U)
    
    return (U, V, J)
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

function frobenius_norm(m1::Matrix{Float64}, m2::Matrix{Float64})
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

function euclidian_distances(v1::Matrix{Float64}, v2)
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

function euclidian_distances_sqrd(v1::Matrix{Float64}, v2::Matrix{Float64})
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

#fuzzy_c_means(data, 2, 1.3)
