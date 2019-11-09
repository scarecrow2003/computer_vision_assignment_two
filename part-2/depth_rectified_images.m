function depth = depth_rectified_images()
    image1 = double(imread('im2.png'));
    image2 = double(imread('im6.png'));
    [height, width, ~] = size(image1);
    nodes_count = width * height;
    h = GCO_Create(nodes_count, 20);
    data_cost = int32(zeros(20, nodes_count));
    max_data_value = 10;
    lambda = 0.04;
    for d = 1:20
        for x = 1:width
            temp = 0;
            if x+d-1 < width
                temp = image2(:, x+d-1, :);
            end
            data_cost(d, x:width:nodes_count) = (sum((image1(:, x, :) - temp).^2, 3))';
        end
    end
    data_cost = lambda * min(data_cost, max_data_value);
    GCO_SetDataCost(h, data_cost);
    d = 1:20;
    D = (d - d').^2;
    max_smooth_cost = 1.7;
    D = min(D, max_smooth_cost);
    GCO_SetSmoothCost(h, D);
    total_edges = (width - 1) * height + (height - 1) * width;
    i = zeros(total_edges, 1);
    j = zeros(total_edges, 1);
    k = 1;
    current = 1;
    for y = 1:height
        for x = 1:width
            node = (y-1) * width + x;
            if x < width
                i(current) = node;
                j(current) = node + 1;
                current = current + 1;
            end
            if y < height
                below = y * width + x;
                i(current) = node;
                j(current) = below;
                current = current + 1;
            end
        end
    end
    neighbour = sparse(i, j, k, nodes_count, nodes_count);
    GCO_SetNeighbors(h, neighbour);
%     GCO_Expansion(h);
    GCO_Swap(h);
    labels = GCO_GetLabeling(h);
    depth = reshape(labels, [height, width]);
    imshow(depth, []);
end