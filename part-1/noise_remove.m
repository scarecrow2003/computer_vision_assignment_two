function img = noise_remove()
    addpath('..')
    source_color = [0, 0, 255]';
    sink_color = [245, 210, 110]';
    input = double(imread('bayes_in.jpg'));
    validate = double(imread('bayes_out.jpg'));
    [height, width, ~] = size(input);
    nodes_count = width * height;
    input = reshape(input, [nodes_count, 3])';
    
    unary = [sum(abs(input - source_color), 1); sum(abs(input - sink_color), 1)] / 3;
    initial_class = double(unary(1, :) >= unary(2, :));
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
    best_result = uint8(zeros(height, width, 3));
    best_lambda = 0;
    for lambda = 1:200
        pairwise = sparse(i, j, lambda, nodes_count, nodes_count);
        label_cost = single([0, 1; 1, 0]);
        [label, ~, ~] = GCMex(initial_class, single(unary), pairwise, label_cost, 0);
        label = reshape(label, [height, width]);
        img = zeros(height, width, 3);
        for y = 1:height
            for x = 1:width
                if label(y, x) == 1
                    img(y, x, :) = sink_color;
                else
                    img(y, x, :) = source_color;
                end
            end
        end
        error = sum(sum(abs(img - validate), 3)/3, 'all')/nodes_count;
        if error < min_error
            min_error = error;
            best_result = img;
            best_lambda = lambda;
        end
        
        % show result of some lambda value
        if lambda == 1 || lambda == 10 || lambda == 50 || lambda == 100 || lambda == 150 || lambda == 200
            imshow(uint8(img));
        end
    end
    disp(int2str(best_lambda));
    imshow(uint8(best_result));
end