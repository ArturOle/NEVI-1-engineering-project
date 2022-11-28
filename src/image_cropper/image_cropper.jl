
using Plots
using Statistics
using Logging
using Distributed
using Colors
using Images
using FileIO

include("clustering/DBSCAN.jl")
include("clustering/FCM.jl")
include("crop.jl")

""" NOTE:
Current implementation has problems with choosing the density for DBSCAN

Possible Solution:
    -   HDBSCAN
    -   Parameter Approximation ( as values are int, pixel location values )

"""

# Minimum size the image can be cropped to (will be variable later)
const MINIMUM_SIZE = (256, 256)


function middle_cluster(number_of_clusters, img_size, data)

    mid = middle(img_size)
    decision_dictionary = Dict{Int, Vector{Float64}}(
        [(x,[]) for x in 1:number_of_clusters]
    )

    for point in data[1]
        append!(
            decision_dictionary[point[end]], 
            [sqrt((mid[1]-point[1][end-1])^2 + (mid[2] - point[1][end])^2)]
        )
    end

    [decision_dictionary[k] = [mean(x)] for (k,x) in decision_dictionary]
    return findmin(decision_dictionary)
end


function findminmax(arr::Vector)

    max = -Inf
    min = Inf
    l = length(arr)
    l = l - l%2

    for i in 1:2:l
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

    return [Int(min), Int(max)]
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

"""
    img_to_graph(image)

Function transforming image into the matrix of points
"""
function image_to_graph(image)

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

    return (img, [max_x, max_y])
end

function get_image(
        image_name::String="ISIC_0024943.jpg",
        directory::String="datasets\\HAM10000\\HAM10000_images"
        )
    
    image = load("$directory\\$image_name")
    resized_image = imresize(image, ratio=1/8)
    original_size = size(image)
    return (resized_image, image, original_size)
end

function resize_cluster_boundries(main_cluster)

    x = [i[2] for i in main_cluster[2]]
    y = [i[1] for i in main_cluster[2]]

    cluster_boundries = [findminmax(y), findminmax(x)]

    for (i, boundry) in enumerate(cluster_boundries)
        cluster_boundries[i] = 8*boundry
    end

    return cluster_boundries
end

function present_results(img, distance_cluster)
    plot(img)
    display(distance_cluster)
    cluster = distance_cluster[2]
    scatter!(
        cluster,
        label="Found cluster", 
        markersize=6,
        markerstrokewidth=0,
        color="#8cffec"
    )
    
    display(current())
end

function stage_information(image_name)
    println(
        """
        ############################################################
        Processing of image $image_name started
        ############################################################
        """
    )
end

function processing(
        image_name::String="ISIC_0024943.jpg";
        directory::String="data\\datasets\\HAM10000",
        number_of_clusters=4,
        m=1.3,
        border=(20,20),
        plot_engine=gr
        )

    stage_information(image_name)
    plot_engine()
    (resized_image, channel_view, original_size) = get_image(image_name, directory)
    (img, img_size) = image_to_graph(resized_image)
        
    data = fuzzy_c_means(img, number_of_clusters, m)
    choosen_cluster = middle_cluster(number_of_clusters, img_size, data)[2]

    db = extract_dimentions(choosen_cluster, data)
    cluster_size = reverse(db[:, end])
    main_cluster = density_clustering(db, cluster_size)
    cluster_boundries = resize_cluster_boundries(main_cluster)

    cropped_image = crop(
        channel_view,
        cluster_boundries,
        original_size,
        minimum_size=MINIMUM_SIZE,
        border=border
    )
    present_results(resized_image, main_cluster)
    return cropped_image
    display(cropped_image)
    
end

function processing(
        quiet::Bool,
        image_name::String="ISIC_0024943.jpg";
        directory::String="data\\datasets\\HAM10000", 
        number_of_clusters=4, 
        m=1.3, 
        border=(10,10)
        )

    (resized_image, original_image, original_size) = get_image(image_name, directory)
    (img, img_size) = image_to_graph(resized_image)
        
    data = fuzzy_c_means(img, number_of_clusters, m, true)
    choosen_cluster = middle_cluster(number_of_clusters, img_size, data)[2]
    db = extract_dimentions(choosen_cluster, data)
    cluster_size = reverse(db[:, end])
    main_cluster = density_clustering(db, cluster_size, true)
    cluster_boundries = resize_cluster_boundries(main_cluster)

    cropped_image = crop(
        original_image, 
        cluster_boundries, 
        original_size, 
        minimum_size=MINIMUM_SIZE, 
        border=border
    )
    
    return imresize(cropped_image, MINIMUM_SIZE)
end

function processing_test()

    n = 24408

    for i in 1:100
        out = processing(true, image_name="ISIC_00$n.jpg")
        save("preprocessed\\img_$i@preprocessed.jpg", out)
        n+=1
    end
end

function process_all()
    io = open("failed.log", "a+")
    logger = Base.SimpleLogger(io)
    Base.with_logger(logger) do
        @info "Starting image processing"
        flush(io)
        files = readdir("datasets\\HAM10000")
        display(files)
        failed_counter = 0
        Threads.@threads for file in files
            try
                out = processing(true, file)
                save("preprocessed\\$file", out)
                flush(io)
            catch ArgumentError
                @error "Image $file failed to be processed"
                failed_counter += 1 
                flush(io)
            end

        end

        @info "Number of failed image processes: $failed_counter"
        flush(io)
    end
    close(io)
end


processing("ISIC_0028249.jpg")
# processing(true, "ISIC_0028328.jpg")
# processing_test()
# process_all()