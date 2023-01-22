using CSV
using Plots
using DataFrames
include("image_cropper/clustering/FCM.jl")

function middle(img_size::Tuple)
    img_size_h = img_size[1]
    img_size_v = img_size[2]

    middle_h = Int64(floor(img_size_h/2))
    middle_v = Int64(floor(img_size_v/2))

    return (middle_h, middle_v)
end

function analysis(csv_data)
    test_csv_1 = Array(csv_data[55, :][1:end-1])
    test_csv_2 = Array(csv_data[92, :][1:end-1])

    image_color_histogram_1 = histogram(test_csv_1, legend=false)
    image_color_histogram_2 = histogram(test_csv_2, legend=false)

    test_csv_1 = reshape(test_csv_1, (28,28))
    test_csv_2 = reshape(test_csv_2, (28,28))

    image_plot_1 = heatmap(test_csv_1, color=:grays)
    image_plot_2 = heatmap(test_csv_2, color=:grays)

    plot(
        image_color_histogram_1, 
        image_color_histogram_2, 
        image_plot_1, 
        image_plot_2, 
        layout=(2, 2)
    )

end

function change_detection(csv_data)
    test_csv_1 = Array(csv_data[55, :][1:end-1])
    #test_csv_2 = Array(csv_data[92, :][1:end-1])

    test_csv_1 = reshape(test_csv_1, (28,28))
    #test_csv_2 = reshape(test_csv_2, (28,28))

    middle_position = middle(size(test_csv_1))

    middle_h = middle_position[1]
    middle_v = middle_position[2]

    println("Horizontal middle: ", middle_h, "\nVertical middle: ", middle_v)
    search_fuzzy_change(test_csv_1) 

end

function search_fuzzy_change(image_data)



end

# function search_change(middle::Tuple, image_data)
#     middle_h = middle[1]
#     middle_v = middle[2]
    
#     horizontal_change_vector = Vector()
#     vertical_change_vector   = Vector()

#     for i in 2:length(image_data[:, middle_h])
#         append!(horizontal_change_vector, (image_data[i, middle_h] - image_data[i-1, middle_h]))
#         append!(vertical_change_vector, (image_data[middle_v, i] - image_data[middle_v, i-1]))
#     end

#     # display(horizontal_change_vector)
#     min_h = findmin(horizontal_change_vector)[2]
#     max_h = findmax(horizontal_change_vector)[2]
    
#     # display(vertical_change_vector)
#     min_v = findmin(vertical_change_vector)[2]
#     max_v = findmax(vertical_change_vector)[2]

#     if min_h - 2 > 0 
#         new_left = min_h - 2 
#     else 
#         new_left = 1
#     end

#     if min_h + 2 <= 28
#         new_right = min_h + 2 
#     else 
#         new_right = 28
#     end

#     if min_v - 2 > 0
#         new_up = min_v -2
#     else 
#         new_up = 1
#     end

#     if min_v + 2 <= 28
#         new_down = min_v + 2
#     else 
#         new_down = 28
#     end
    
#     display((new_left, new_right))
#     display((new_down, new_up))

#     new_image = extract(image_data, (new_left, new_right), (new_up, new_down))

#     # new_image = @view image_data[3 4 ;3 4]
#     #display(new_image)
#     heatmap(new_image, color=:grays)
# end

function extract(matrix::Matrix, x_range::Tuple, y_range::Tuple)

    extracted = zeros(y_range[2]-y_range[1]+1, x_range[2]-x_range[1]+1)

    for i in x_range[1]:x_range[2]
        for j in y_range[1]:y_range[2]
            extracted[j-y_range[1]+1,i-x_range[1]+1] = matrix[:, i][j]
        end
    end

    return extracted
end

function chop_images(filename="./datasets/HAM10000/hmnist_28_28_L.csv")
    csv_data = CSV.read(filename, DataFrame)

    change_detection(csv_data)

end

chop_images()

# matrix = [
#     1 2 3 4;
#     2 3 4 1;
#     3 4 1 2;
#     4 1 2 3   
# ]

# extract(matrix, (2, 3), (1, 4))

