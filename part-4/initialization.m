function result = initialization()
    addpath('..');
    addpath('Road');
    addpath(['Road' filesep 'src']);
    image = double(imread('test0000.jpg'));
    [height, width, ~] = size(image);
    nodes_count = width * height;
    
    file = fopen(['Road' filesep 'cameras.txt'], 'r');
    cameras = fscanf(file, '%f %f %f', [3, Inf]);
    fclose(file);
    
    min_displacement = 0;
    max_displacement = 0.01;
    step = 0.0001;
    displacement = 1 + (max_displacement - min_displacement) / step;
    
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
    
    epsilon = 50;
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
    w_s = 5 ./ (max_displacement - min_displacement);

    [x, y] = meshgrid(1:displacement, 1:displacement);
    eta = (max_displacement - min_displacement) * 0.05;
    label_cost = single(min(step .* abs(x - y), eta));
    
    [x, y] = meshgrid(1:width, 1:height);
    location1 = [x(:)'; y(:)'; ones(1, nodes_count)];
    sigma_c = 10;
    
    num = 3;
    for n_current = num:140-num
        image1 = double(imread([['Road' filesep 'src' filesep 'test'], sprintf('%04d', n_current),'.jpg']));
        camera_start = n_current * 7;
        k1 = cameras(:, 1+camera_start:3+camera_start)';
        r1 = cameras(:, 4+camera_start:6+camera_start)';
        t1 = cameras(:, 7+camera_start);
        
        u_init = zeros(displacement, nodes_count);
        for s = [n_current-3, n_current-2, n_current-1, n_current+1, n_current+2, n_current+3]
            image2 = double(imread([['Road' filesep 'src' filesep 'test'], sprintf('%04d', s), '.jpg']));
            camera_start = s * 7;
            k2 = cameras(:, 1+camera_start:3+camera_start)';
            r2 = cameras(:, 4+camera_start:6+camera_start)';
            t2 = cameras(:, 7+camera_start);

            pixel1 = impixel(image1, location1(1,:), location1(2,:));
            u = zeros(displacement, nodes_count);
            for d = 1:displacement
                location2 = k2 * r2' * r1 / k1 * location1 + k2 * r2' * (t1 - t2) * step * (d - 1);
                location2 = round(location2 ./ location2(3, :));
                pixel2 = impixel(image2, location2(1, :), location2(2, :));
                pixel2(isnan(pixel2)) = 0;
                u(d, :) = (sigma_c ./ (sigma_c + sqrt(sum((pixel1 - pixel2) .^2, 2))))';
            end
            u_init = u_init + u;
        end
        unary = 1 - u_init ./ max(u_init);
        [~, class] = min(unary); 
        class = class - 1;
        
        image1 = reshape(image1, 1, nodes_count, 3);
        lambda = 1 ./ (sqrt(sum((image1(1, i, :) - image1(1, j, :)) .^ 2, 3)) + epsilon);
        pairwise = sparse(i, j, lambda);
        mu = n ./ full(sum(pairwise));
        lambda = w_s .* lambda .* mu(i);
        pairwise = sparse(i, j, lambda);
        
        [label, ~, ~] = GCMex(class, single(unary), pairwise, label_cost, 1);
        label = reshape(label , [height, width]);
        result = mat2gray(label);
        imwrite(result, [['initialization' filesep 'test'], sprintf('%04d', n_current), '.jpg']);
        save([['initialization' filesep 'test'], sprintf('%04d', n_current), '.mat'], 'label');
    end
end