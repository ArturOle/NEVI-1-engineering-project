

function range_check_smaller(calculated_range, projected_size, minimum_size, img_size)

    display("#############")
    display(calculated_range)
    display(projected_size)
    display(img_size)
    display("#############")

    if projected_size[1] < minimum_size[1]
        x_move = Int(ceil((minimum_size[1] - projected_size[1])/2))

        x_lower = calculated_range[1][1] - x_move
        x_higher = calculated_range[1][2] + x_move

        if x_lower < 1
            x_lower = calculated_range[1][1]
            x_higher = calculated_range[1][2] + x_move*2
        elseif x_higher >= img_size[1]
            x_lower = calculated_range[1][1] - x_move*2
            x_higher = calculated_range[1][2] 
        end

        calculated_range = (
            (x_lower, x_higher),
            (calculated_range[2][1], calculated_range[2][2])
        )
        display(calculated_range)
    end

    if projected_size[2] < minimum_size[2]
        y_move = Int(ceil(( minimum_size[2] - projected_size[2])/2))

        y_lower = calculated_range[2][1] - y_move
        y_higher = calculated_range[2][2] + y_move

        if y_lower < 1
            y_lower = calculated_range[2][1]
            y_higher = calculated_range[2][2] + y_move*2
        elseif y_higher >= img_size[2]
            y_lower = calculated_range[2][1] - y_move*2
            y_higher = calculated_range[2][2] 
        end

        calculated_range = (
            (calculated_range[1][1], calculated_range[1][2]),
            (y_lower, y_higher)
        )
        display(calculated_range)
    end

    return calculated_range
end
"""
function crop(img, initial_range, img_size, minimum_size=(256, 256), border=(20, 20))
    # Size of image based on the acquired points
    if initial_range[1][2] > img_size[1]
        initial_range[1][2] = img_size[1]
    end
    if initial_range[2][2] > img_size[2]
        initial_range[2][2] = img_size[2]
    end
    projected_size = (
        initial_range[1][2]-initial_range[1][1], 
        initial_range[2][2]-initial_range[2][1]
    )
    calculated_range = initial_range
    # calculated_range = ()
    if projected_size < minimum_size
        println("projected_size is smaller than minimum_size")
    
        if projected_size[1] < minimum_size[1]
            x_move = Int(ceil((minimum_size[1] - projected_size[1])/2))

            x_lower = calculated_range[1][1] - x_move
            x_higher = calculated_range[1][2] + x_move

            if x_lower < 1
                x_lower = calculated_range[1][1]
                x_higher = calculated_range[1][2] + x_move*2
            elseif x_higher >= img_size[1]
                x_lower = calculated_range[1][1] - x_move*2
                x_higher = calculated_range[1][2] 
            end

            calculated_range = (
                (x_lower, x_higher),
                (calculated_range[2][1], calculated_range[2][2])
            )
            display(calculated_range)
        end

        if projected_size[2] < minimum_size[2]
            y_move = Int(ceil(( minimum_size[2] - projected_size[2])/2))

            y_lower = calculated_range[2][1] - y_move
            y_higher = calculated_range[2][2] + y_move

            if y_lower < 1
                y_lower = calculated_range[2][1]
                y_higher = calculated_range[2][2] + y_move*2
            elseif y_higher >= img_size[2]
                y_lower = calculated_range[2][1] - y_move*2
                y_higher = calculated_range[2][2] 
            end

            calculated_range = (
                (calculated_range[1][1], calculated_range[1][2]),
                (y_lower, y_higher)
            )
            display(calculated_range)
        end

    elseif projected_size > img_size
        println("projected_size is greated than img_size")

        calculated_range = (
            (1, img_size[1]),
            (1, img_size[2])
        )

    else
        println("projected_size is greated than minimum_size")

        x_lower = calculated_range[1][1]-border[1]
        x_higher = calculated_range[1][2]+border[1]
        y_lower = calculated_range[2][1]-border[2]
        y_higher = calculated_range[2][2]+border[2]

        if x_lower > 0 && x_higher < img_size[1] && y_lower > 0 && y_higher < img_size[2]
            calculated_range = (
                (x_lower, x_higher),
                (y_lower, y_higher)
            )
        else
            calculated_range = (
                (1, img_size[1]),
                (1, img_size[2])
            )
        end
    end

    img = img[calculated_range[2][1]:calculated_range[2][2], calculated_range[1][1]:calculated_range[1][2]]
    return img
end

"""

return_original_size(img_size) = ((1, img_size[1]), (1, img_size[2]))


function range_check(calculated_range, projected_size, minimum_size, img_size, border)

    if projected_size < minimum_size
        println("$projected_size is smaller than $minimum_size")
        calculated_range = range_check_smaller(calculated_range, projected_size, minimum_size, img_size)
        
        #calculated_range = add_border(calculated_range, border, img_size, true)
        return calculated_range

    elseif projected_size > img_size
        println("$projected_size is greater than $img_size")

        return return_original_size(img_size)
        
    else
        println("$projected_size is greated than $minimum_size")
        
        return add_border(calculated_range, border, img_size)

    end
end

function add_border(calculated_range, border, img_size, use_range::Bool)
    x_lower = calculated_range[1][1]-border[1]
    x_higher = calculated_range[1][2]+border[1]
    y_lower = calculated_range[2][1]-border[2]
    y_higher = calculated_range[2][2]+border[2]

    if x_lower > 0 && x_higher < img_size[1] && y_lower > 0 && y_higher < img_size[2]
        return ((x_lower, x_higher), (y_lower, y_higher))
    else
        return calculated_range
    end
end

function add_border(calculated_range, border, img_size)
    x_lower = calculated_range[1][1]-border[1]
    x_higher = calculated_range[1][2]+border[1]
    y_lower = calculated_range[2][1]-border[2]
    y_higher = calculated_range[2][2]+border[2]

    if x_lower > 0 && x_higher < img_size[1] && y_lower > 0 && y_higher < img_size[2]
        return ((x_lower, x_higher), (y_lower, y_higher))
    else
        return return_original_size(img_size)
    end
end

function crop(img, initial_range, img_size; minimum_size=(256, 256), border=(20, 20))

    # Size of image based on the acquired points
    if initial_range[1][2] > img_size[1]
        initial_range[1][2] = img_size[1]
    end

    if initial_range[2][2] > img_size[2]
        initial_range[2][2] = img_size[2]
    end

    projected_size = (
        initial_range[1][2]-initial_range[1][1], 
        initial_range[2][2]-initial_range[2][1]
    )
    display("Here $projected_size")
    calculated_range = range_check(
        initial_range, 
        projected_size, 
        minimum_size, 
        img_size, 
        border
    )
    display(calculated_range)
    img = img[
        calculated_range[2][1]:calculated_range[2][2], 
        calculated_range[1][1]:calculated_range[1][2]
    ]
    return img
end
