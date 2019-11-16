function result = depth_from_stereo()
    addpath('..');
    image1 = double(imread('test00.jpg'));
    image2 = double(imread('test09.jpg'));
    [height, width, ~] = size(image1);
    nodes_count = width * height;
    
    file = fopen('cameras.txt', 'r');
    cameras = fscanf(file, '%f %f %f', [3, 14]);
    fclose(file);
    k1 = cameras(:, 1:3)';
    r1 = cameras(:, 4:6)';
    t1 = cameras(:, 7);
    k2 = cameras(:, 8:10)';
    r2 = cameras(:, 11:13)';
    t2 = cameras(:, 14);
    
    min_displacement = 0;
    max_displacement = 0.01;
    step = 0.0001;
    displacement = 1 + (max_displacement - min_displacement) / step;

    [x, y] = meshgrid(1:width, 1:height);
    location1 = [x(:)'; y(:)'; ones(1, nodes_count)];
    pixel1 = impixel(image1, location1(1, :), location1(2, :));
    unary = zeros(displacement, nodes_count);
    sigma_c = 10;
    for d = 1:displacement
        location2 = k2 * r2' * r1 / k1 * location1 + k2 * r2' * (t1 - t2) * step * (d - 1);
        location2 = round(location2 ./ location2(3, :)); % normalize
        pixel2 = impixel(image2, location2(1, :), location2(2, :));
        pixel2(isnan(pixel2)) = 0;
        unary(d, :) = (sigma_c ./ (sigma_c + sqrt(sum((pixel1 - pixel2) .^ 2, 2))))';
    end
    unary = 1 - (unary ./ max(unary));
    [~, class] = min(unary); 
    class = class - 1;
    
    total_edges = (width - 1) * height + (height - 1) * width;
    i = zeros(total_edges, 1);
    j = zeros(total_edges, 1);
    current = 1;
    for x = 1:width
        for y = 1:height
            n_current = (x - 1) * height + y;
            if y < height
                i(current) = n_current;
                j(current) = n_current + 1;
                current = current + 1;
            end
            if y > 1
                i(current) = n_current;
                j(current) = n_current - 1;
                current = current + 1;
            end
            if x < width
                i(current) = n_current;
                j(current) = n_current + height;
                current = current + 1;
            end
            if x > 1
                i(current) = n_current;
                j(current) = n_current - height;
                current = current + 1;
            end
        end
    end
    
    image1 = reshape(image1, 1, nodes_count, 3);
    epsilon = 50;
    lambda = 1 ./ (sqrt(sum((image1(1, i, :) - image1(1, j, :)) .^ 2, 3)) + epsilon);
    pairwise = sparse(i, j, lambda);
    n = 4 .* ones(height, width);
    n(:, 1) = 3;
    n(:, end) = 3;
    n(1, :) = 3;
    n(end, :) = 3;
    n(1, 1) = 2;
    n(1, end) = 2;
    n(end, 1) = 2;
    n(end, end) = 2;
    n = reshape(n, 1, nodes_count);
    mu = n ./ full(sum(pairwise));
    w_s = 5 ./ (max_displacement - min_displacement);
    lambda = w_s .* lambda .* mu(i);
    pairwise = sparse(i, j, lambda);

    [x, y] = meshgrid(1:displacement, 1:displacement);
    eta = (max_displacement - min_displacement) * 0.05;
    label_cost = single(min(step .* abs(x - y), eta));

    [label, ~, ~] = GCMex(class, single(unary), pairwise, label_cost, 1);
    label = reshape(label , [height, width]);
    result = mat2gray(label);
    imshow(result)
end