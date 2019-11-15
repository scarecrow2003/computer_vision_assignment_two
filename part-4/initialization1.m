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
            n = (x - 1) * height + y;
            if y < height
                i(current) = n;
                j(current) = n + 1;
                current = current + 1;
            end
            if y > 1
                i(current) = n;
                j(current) = n - 1;
                current = current + 1;
            end
            if x < width
                i(current) = n;
                j(current) = n + height;
                current = current + 1;
            end
            if x > 1
                i(current) = n;
                j(current) = n - height;
                current = current + 1;
            end
        end
    end
    
    eta=0.05.*(max_displacement-min_displacement);
    epsilon=50;
    ws=5./(max_displacement-min_displacement);
    sigmac=10;
    sigmad=2.5;
    nei_num=3;
    iter_num=2;

    

    nei=4.*ones(height,width);
    nei(:,1)=3;
    nei(:,end)=3;
    nei(1,:)=3;
    nei(end,:)=3;
    nei(1,1)=2;
    nei(1,end)=2;
    nei(end,1)=2;
    nei(end,end)=2;
    nei=reshape(nei,1,nodes_count);

    [label_X,label_Y]=meshgrid(1:displacement,1:displacement);
    labelcost=min(step.*abs(label_X-label_Y),eta);

    [X,Y]=meshgrid(1:width,1:height);
    loc_cen=[X(:)';Y(:)';ones(1,nodes_count)];

    for n=nei_num:140-nei_num
        img_cen=double(imread([['Road' filesep 'src' filesep 'test'],sprintf('%04d',n),'.jpg']));
        img_cen_prime=reshape(img_cen,1,nodes_count,3);

        seq=n*7;
        K_cen=cameras(:,1+seq:3+seq)';
        R_cen=cameras(:,4+seq:6+seq)';
        T_cen=cameras(:,7+seq);

        lambda=1./(sqrt(sum((img_cen_prime(1,i,:)-img_cen_prime(1,j,:)).^2,3))+epsilon);
        prior=sparse(i,j,lambda);
        u=nei./full(sum(prior));
        lambda=ws.*lambda.*u(i);
        pairwise=sparse(i,j,lambda);

        L_init=zeros(displacement,nodes_count);
        for b=[n-3,n-2,n-1,n+1,n+2,n+3]
            img_nei=double(imread([['Road' filesep 'src' filesep 'test'],sprintf('%04d',b),'.jpg']));
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
        imwrite(result,[['initialization' filesep 'test'],sprintf('%04d',n),'.jpg']);
        save([['initialization' filesep 'test'],sprintf('%04d',n),'.mat'],'label');
    end

end