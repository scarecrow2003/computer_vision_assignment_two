function result = initialization1()
    addpath('..');
    addpath('Road');
    addpath(['Road' filesep 'src']);
    image = double(imread('test0000.jpg'));
    [height, width, ~] = size(image);
    nodes_count = height * width;
    
    file = fopen(['Road' filesep 'cameras.txt'],'r');
    cameras = fscanf(file, '%f %f %f', [3,Inf]);
    fclose(file);
    
    min_displacement = 0;
    max_displacement = 0.01;
    step = 0.0001;
    displacement = 1 + (max_displacement - min_displacement) / step;
    
    total_edges = (width - 1) * height + (height - 1) * width;
    i = zeros(total_edges, 1);
    j = zeros(total_edges, 1);
    current = 1;
    for x = 1:width
        for y = 1:height
            node = (x - 1) * height + y;
            if y < height
                i(current) = node;
                j(current) = node + 1;
                current = current + 1;
            end
            if y > 1
                i(current) = node;
                j(current) = node - 1;
                current = current + 1;
            end
            if x < width
                i(current) = node;
                j(current) = node + height;
                current = current + 1;
            end
            if x > 1
                i(current) = node;
                j(current) = node - height;
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
    n = reshape(n, 1,nodes_count);
    
    eta=0.05.*(max_displacement-min_displacement);
    
    ws=5./(max_displacement-min_displacement);
    sigmac=10;
    sigmad=2.5;
    nei_num=3;
    iter_num=2;

    

    

    [label_X,label_Y]=meshgrid(1:displacement,1:displacement);
    labelcost=min(step.*abs(label_X-label_Y),eta);

    [X,Y]=meshgrid(1:width,1:height);
    loc_cen=[X(:)';Y(:)';ones(1,nodes_count)];

    for n_current=nei_num:140-nei_num
        img_cen=double(imread(['.\Assignment2_A0186492R_YaoYuan\Road\src\test',sprintf('%04d',n_current),'.jpg']));
        img_cen_prime=reshape(img_cen,1,nodes_count,3);

        seq=n_current*7;
        K_cen=cameras(:,1+seq:3+seq)';
        R_cen=cameras(:,4+seq:6+seq)';
        T_cen=cameras(:,7+seq);

        lambda=1./(sqrt(sum((img_cen_prime(1,i,:)-img_cen_prime(1,j,:)).^2,3))+epsilon);
        prior=sparse(i,j,lambda);
        u=n./full(sum(prior));
        lambda=ws.*lambda.*u(i);
        pairwise=sparse(i,j,lambda);

        L_init=zeros(displacement,nodes_count);
        for b=[n_current-3,n_current-2,n_current-1,n_current+1,n_current+2,n_current+3]
            img_nei=double(imread(['.\Assignment2_A0186492R_YaoYuan\Road\src\test',sprintf('%04d',b),'.jpg']));
            seq=b*7;
            K_nei=cameras(:,1+seq:3+seq)';
            R_nei=cameras(:,4+seq:6+seq)';
            T_nei=cameras(:,7+seq);

            pix_cen=impixel(img_cen,loc_cen(1,:),loc_cen(2,:));
            for dis=1:displacement
                loc_nei=K_nei*R_nei'*R_cen/K_cen*loc_cen+K_nei*R_nei'*(T_cen-T_nei).*step.*(dis-1);
                loc_nei=round(loc_nei./loc_nei(3,:));
                pix_nei=impixel(img_nei, loc_nei(1,:), loc_nei(2,:));
                pix_nei(isnan(pix_nei))=0;
                pc(dis,:)=(sigmac./(sigmac+sqrt(sum((pix_cen-pix_nei).^2,2))))';
            end
            L_init=L_init+pc;
        end
        unary=1-L_init./max(L_init);
        [~,segclass]=min(unary); 
        segclass=segclass-1;
        [label,~,~] = GCMex(segclass,single(unary),pairwise,single(labelcost),1);
        label=reshape(label,height,width);
        result=mat2gray(label);
        imwrite(result,['.\Assignment2_A0186492R_YaoYuan\PART4_initialization\test',sprintf('%04d',n_current),'.jpg']);
        save(['.\Assignment2_A0186492R_YaoYuan\PART4_initialization\test',sprintf('%04d',n_current),'.mat'],'label');
    end

end