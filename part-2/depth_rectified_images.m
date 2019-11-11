function result = depth_rectified_images()
    addpath('..')
    image1 = double(imread('im2.png'));
    image2 = double(imread('im6.png'));
    [height, width, ~] = size(image1);
    nodes_count = width * height;
    image1 = reshape(image1, 1, nodes_count, 3);
    image2 = reshape(image2, 1, nodes_count, 3);
    
    depth = rgb2gray(im2double(imread('depth.png')));
    [d_height, d_width] = size(depth);
    depth = resample(depth', width, d_width);
    depth = resample(depth', height, d_height);
    
    d = 1:80;
    unary = zeros(80, nodes_count);
    for disparity = d
        displacement = (disparity - 40) * height;
        if displacement < 0
            unary(disparity, :) = sum(abs(cat(2, image2(1, -displacement+1:nodes_count, :), zeros(1, -displacement, 3)) - image1), 3) / 3;
        else
            unary(disparity, :) = sum(abs(cat(2, zeros(1, displacement, 3), image2(1, 1:nodes_count - displacement, :)) - image1), 3) / 3;
        end
    end
    
    [~, class] = min(unary); 
    class = class - 1;
    
    label_cost = single(log(1 + ((d - d').^2) / 2));
    
    total_edges = (width - 1) * height + (height - 1) * width;
    i = zeros(total_edges, 1);
    j = zeros(total_edges, 1);
    current = 1;
    for y = 1:height
        for x = 1:width
            node = (y-1) * width + x;
            if x < width
                i(current) = node;
                j(current) = node + 1;
                current = current + 1;
            end
            if x > 1
                i(current) = node;
                j(current) = node - 1;
                current = current + 1;
            end
            if y < height
                below = y * width + x;
                i(current) = node;
                j(current) = below;
                current = current + 1;
            end
            if y > 1
                above = (y-1) * width + x;
                i(current) = node;
                j(current) = above;
                current = current + 1;
            end
        end
    end
    
    min_error = 10000;
    result = uint8(zeros(height, width, 3));
    best_lambda = 0;
    for lambda = 1:200
        pairwise = sparse(i, j, lambda, nodes_count, nodes_count);
        [label, ~, ~] = GCMex(class, single(unary), pairwise, label_cost, 1);
        label = reshape(label, [height, width]);
        label = mat2gray(label);
        error = sum(abs(label - depth), 'all') / nodes_count;
        if error < min_error
            min_error = error;
            result = label;
            best_lambda = lambda;
        end
    end
    disp(int2str(best_lambda));
    imshow(result);
end