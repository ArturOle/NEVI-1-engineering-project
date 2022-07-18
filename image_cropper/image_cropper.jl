
using Plots
using Statistics

using Colors
using Images
using FileIO

include("DBSCAN.jl")
include("FCM.jl")
include("crop.jl")

""" NOTE:
Current implementation has problems with choosing the densitu for DBSCAN

Possible Solution:
    -   HDBSCAN
    -   Parameter Approximation ( as values are int, pixel location values )

"""

# Minimum size the image can be cropped to 
const MINIMUM_SIZE = (256, 256)

function img_to_graph(image_name::String="HAM10000_images\\ISIC_0024944.jpg")
    """
    Function transforming image into the matrix of points  
    """
    # load and convert to color matrix
    image = load("datasets\\HAM10000\\$image_name")
    cv = channelview(image)
    s = size(cv)
    max_x = s[3]
    max_y = s[2]

    # transform to data in a form of vectors of every dimention
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


function middle_cluster(number_of_clusters, img_size, data)
    mid = middle(img_size)
    decision_dictionary = Dict{Int, Vector{Float64}}(
        [[x,[]] for x in 1:number_of_clusters]
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

function image_to_graph(image)
    cv = channelview(image)
    img = zeros(Float64, 5, max_x*max_y)
    s = size(cv)

    max_x = s[3]
    max_y = s[2]

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
    return (resized_image, original_size)
end

function resize_cluster_boundries(main_cluster)
    x = [i[2] for i in main_cluster[2]]
    y = [i[1] for i in main_cluster[2]]

    cluster_boundries = [findminmax(y), findminmax(x)]
    display(cluster_boundries)

    for i in 1:length(cluster_boundries)
        cluster_boundries[i] = 8*cluster_boundries[i]
    end
end

function processing(
        ;image_name::String="ISIC_0024943.jpg",
        directory::String="datasets\\HAM10000\\HAM10000_images",
        number_of_clusters=4,
        m=1.3,
        border=(10,10),
        plot_engine=gr)

    println(
        """
        ############################################################
        Processing of image $image_name started
        ############################################################
        """
    )
    plot_engine()
    (resized_image, original_size) = get_image(image_name, directory)
    (img, img_size) = image_to_graph(resized_image)
        
    data = fuzzy_c_means(img, number_of_clusters, m)
    choosen_cluster = middle_cluster(number_of_clusters, img_size, data)[2]

    db = extract_dimentions(choosen_cluster, data)
    cluster_size = reverse(db[:, end])
    main_cluster = density_clustering(db, cluster_size)
    cluster_boundries = resize_cluster_boundries(main_cluster)

    cropped_image = crop(
        image,
        cluster_boundries,
        original_size,
        minimum_size=MINIMUM_SIZE,
        border=border
    )
    return cropped_image
end

function processing(
        quiet::Bool; 
        image_name::String="ISIC_0024943.jpg", 
        directory::String="datasets\\HAM10000\\HAM10000_images", 
        number_of_clusters=4, 
        m=1.3, 
        border=(10,10)
        )

    (resized_image, original_size) = get_image(image_name, directory)
    (img, img_size) = image_to_graph(resized_image)
        
    data = fuzzy_c_means(img, number_of_clusters, m, true)
    choosen_cluster = middle_cluster(number_of_clusters, img_size, data)[2]

    db = extract_dimentions(choosen_cluster, data)
    cluster_size = reverse(db[:, end])
    main_cluster = density_clustering(db, cluster_size, true)
    cluster_boundries = resize_cluster_boundries(main_cluster)

    cropped_image = crop(
        image, 
        cluster_boundries, 
        original_size, 
        minimum_size=MINIMUM_SIZE, 
        border=border
    )
    return cropped_image
end

function processing_test()
    n = 24408

    for i in 1:100
        out = processing(true, image_name="ISIC_00$n.jpg")
        save("preprocessed\\img_$i.jpg", out)
        n+=1
    end
end

function process_all()
    files = readdir("datasets\\HAM10000\\HAM10000_images")
    display(files)

    for file in files
        out = processing(true, image_name="ISIC_00$n.jpg")
        save("preprocessed\\$file", out)
    end

    # for i in 1:100
    #     out = processing(true, "HAM10000_images\\ISIC_00$n.jpg")
    #     save("preprocessed\\img_$i.jpg", out)
    #     n+=1
    # end

end


# processing("HAM10000_images\\ISIC_0028245.jpg")
# # processing(true, "HAM10000_images\\ISIC_0028339.jpg")
# # processing_test()
