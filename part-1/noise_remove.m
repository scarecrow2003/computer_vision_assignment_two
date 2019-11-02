function img = noise_remove(lambda)
    source_color = uint8([0, 0, 255])';
    sink_color = uint8([245, 210, 110])';
    input = imread('bayes_in.jpg');
    [height, width, ~] = size(input);
    nodes_count = width * height;
    input = reshape(input, [nodes_count, 3])';

    h = GCO_Create(nodes_count, 2);
    data_cost = int32([sum(abs(input - source_color), 1); sum(abs(input - sink_color), 1)] / 3);
    GCO_SetDataCost(h, data_cost);
    total_edges = (width - 1) * height + (height - 1) * width;
    i = zeros(total_edges, 1);
    j = zeros(total_edges, 1);
    k = lambda;
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
    GCO_Expansion(h);
    labels = GCO_GetLabeling(h);
    labels = reshape(labels, [height, width]);
    img = uint8(zeros(height, width, 3));
    for y = 1:height
        for x = 1:width
            if labels(y, x) == 1
                img(y, x, :) = source_color;
            else
                img(y, x, :) = sink_color;
            end
        end
    end
    imshow(img);
end