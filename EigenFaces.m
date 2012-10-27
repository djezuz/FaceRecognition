clc;
clear all;
[filename1, pathname1] = uigetfile({'*.*'},'Get Database');
load(filename1)

starting_individual=1;%maximum value = 40
ending_individual=5;
number_of_individuals=ending_individual-starting_individual+1;%maximum value = 40

starting_images_per_individual=1;%maximum value = 10
ending_images_per_individual=5;%maximum value = 10
number_of_images_per_individual=ending_images_per_individual-starting_images_per_individual+1;

number_total_images=number_of_individuals*number_of_images_per_individual;

sample=zeros(4096,number_total_images);
for i=starting_individual:ending_individual
    for j=starting_images_per_individual:ending_images_per_individual
              sample(:,(i-1)*number_of_individuals+j)=faces(:,(i-1)*10+j);
    end
end
% 
% %standardizing the images:
% for i=1:number_total_images
%     meanpixel=mean(sample(:,i));
%     stdpixel=std(sample(:,i));
%     sample(:,i)=(sample(:,i)-meanpixel)*80/stdpixel+100;
%     
%     figure(1);
%     Image=reshape(sample(:,i),64,64);
%     image(Image)
%     colormap(gray(256));
%     daspect([1 1 1]);
%     pause(0.2);
%     
% 
% end%worsens performance as of now. 




avg=zeros(4096,1);
total=zeros(4096,1);

    for i=1:number_total_images
        total=total+sample(:,i);
    end
    


avg=total/number_total_images;

    Image=reshape(avg,64,64);
    imagesc(Image);
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Mean Image');
    title(str,'FontSize',12);
    pause(0.2);
%     saveas(gcf,'Report PCA/mean.jpg');
    
newset=zeros(4096,number_total_images);
for i=1:number_total_images
    newset(:,i)=sample(:,i)-avg;
end


%%


Cov=(newset'*newset)./number_total_images;
[ev v]=eig(Cov);
% 
[v,ind]=sort(diag(v),'descend')
ev=ev(:,ind);


for i=1:number_total_images
    temp=sqrt(sum(ev(:,i).^2));
    ev(:,i)=ev(:,i)./temp;
end

for i=1:number_total_images
    Eimage(:,i)=newset*ev(:,i)/(v(i));
end

for i=1:number_total_images
    temp=sqrt(sum(Eimage(:,i).^2));
    Eimage(:,i)=Eimage(:,i)./temp;
end


% for i=1:number_total_images
%     EImages(:,i)=(newset*one(:,i))/(v(i,1)*number_total_images);
% end


for i=1:10
    subplot(2,5,i); 
    Image=reshape(Eimage(:,i),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    pause(0.1);
    axis off;
    str=sprintf('Eigen image %d',i);
    title(str,'FontSize',10);
end
% str=sprintf('Report PCA/Eigenimage%d.jpg',i);
saveas(gcf,'Report PCA/Eigenimage.jpg');

total=0;
for i=1:number_total_images
    total=total+v(i);
end


dcap=0;
for i=1:number_total_images
    newsum=0;
    for j=1:i
        newsum=newsum+v(j);
    end
    
    if(newsum > 0.95*total)
        dcap=i;
        break;
    end
    
end
dcap

% i=1:number_total_images;

figure(2);
plot(1:dcap,v(1:dcap),'+');
hold on;
plot(dcap+1:number_total_images,v(dcap+1:number_total_images),'o');
xlabel('index');
ylabel('EigenValue');

saveas(gcf,'Report PCA/Plot of eigen values in desceding order.jpg');

%%

reconDcap=zeros(4096,number_total_images);
reconPure=zeros(4096,number_total_images);
rmse=zeros(number_total_images,1);
WeightSet=zeros(dcap,number_total_images);

for imagecount=1:number_total_images

%% Pure reconstruction

InputImage=sample(:,imagecount);
temp=InputImage;
% m=mean(temp);
% st=std(temp);
% temp=(temp-m)*80/st+100;
NormImage = temp;
% Difference = temp-m;
    
ProjectionPure=zeros(number_total_images,1);    
for i=1:number_total_images
    ProjectionPure(i,1)=dot(Eimage(:,i),NormImage-avg);
end

% Projection=Projection/norm(Projection);
% norm(Projection)
reconPure(:,imagecount)=Eimage*ProjectionPure+avg;

    figure(1);
    subplot(1,3,1);
    Image=reshape(NormImage,64,64);
    image(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Original Image');
    title(str,'FontSize',12);

    
    subplot(1,3,2);
    Image=reshape(reconPure(:,imagecount),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Reconstructed image \n using all eigen vectors');
    title(str,'FontSize',12);
    

%% Reconstruction with dcap

% dcap=24;
% dcap= 25;

BasisEVector=Eimage(:,1:dcap);
ProjectionDcap=zeros(dcap,1);

for i=1:dcap
    ProjectionDcap(i,1)=dot(BasisEVector(:,i),NormImage-avg);
    
end

WeightSet(:,imagecount)=ProjectionDcap;
reconDcap(:,imagecount)=BasisEVector*ProjectionDcap+avg;
total=0;
for j=1:4096
    total=sum(reconDcap(j,imagecount)-sample(j,imagecount)).^2;
end
rmse(imagecount,1)=sqrt(total/4096);

    
    subplot(1,3,3);
    Image=reshape(reconDcap(:,imagecount),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    pause(0.2);
    str=sprintf('Reconstructed image.\n Dcap = %d\nRMSE=%f',dcap,rmse(imagecount,1));
    title(str,'FontSize',12);
    
    str=sprintf('Report PCA/ReconstructionImage%d.jpg',imagecount);
    saveas(gcf,str);
    

end




%% Testing
% take a new set of images of the same individuals to test. 


test_starting_individual=1;%maximum value = 40
test_ending_individual=5;
test_number_of_individuals=test_ending_individual-test_starting_individual+1;%maximum value = 400

test_starting_images_per_individual=6;%maximum value = 10
test_ending_images_per_individual=10;%maximum value = 10
test_number_of_images_per_individual=test_ending_images_per_individual-test_starting_images_per_individual+1;

test_number_total_images=test_number_of_individuals*test_number_of_images_per_individual;
test=zeros(4096,test_number_total_images);

imagecount=1;
for i=test_starting_individual:test_ending_individual
    for j=test_starting_images_per_individual:test_ending_images_per_individual
        test(:,imagecount)=faces(:,(i-1)*10+j);
        imagecount=imagecount+1
    end
end

reconPureTest=zeros(4096,test_number_total_images);
reconDcapTest=zeros(4096,test_number_total_images);
rmsetest=zeros(test_number_total_images,1);

for imagecount=1:test_number_total_images

%% Pure reconstruction
InputImage=test(:,imagecount);
temp=InputImage;
% m=mean(temp);
% st=std(temp);
% temp=(temp-m)*80/st+100;
NormImage = temp;
% Difference = temp-m;
    
ProjectionPureTest=zeros(test_number_total_images,1);    
for i=1:test_number_total_images
    ProjectionPureTest(i,1)=dot(Eimage(:,i),NormImage-avg);
end

% Projection=Projection/norm(Projection);
% norm(Projection)
reconPureTest(:,imagecount)=Eimage*ProjectionPureTest+avg;

    figure(1);
    subplot(1,3,1);
    Image=reshape(NormImage,64,64);
    image(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Original Image');
    title(str,'FontSize',12);

    
    subplot(1,3,2);
    Image=reshape(reconPureTest(:,imagecount),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Reconstructed image \n using all eigen vectors');
    title(str,'FontSize',12);
    

%% Reconstruction with dcap

% dcap=24;
% dcap= 25;

BasisEVector=Eimage(:,1:dcap);
ProjectionDcapTest=zeros(dcap,1);

for i=1:dcap
    ProjectionDcapTest(i,1)=dot(BasisEVector(:,i),NormImage-avg);
end

% Projection=Projection/norm(Projection);
% norm(Projection)
reconDcapTest(:,imagecount)=BasisEVector*ProjectionDcapTest+avg;

total=0;
for j=1:4096
    total=sum(reconDcapTest(j,imagecount)-test(j,imagecount)).^2;
end
rmsetest(imagecount)=sqrt(total/4096);
    
    subplot(1,3,3);
    Image=reshape(reconDcapTest(:,imagecount),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    pause(0.2);
    str=sprintf('Reconstructed image.\n Dcap = %d \n RMSE=%f',dcap,rmsetest(imagecount));
    title(str,'FontSize',12);
    
    str=sprintf('Report PCA/ReconstructionImageOfDiffImages%d.jpg',imagecount);
    saveas(gcf,str);
    

end



%% Testing for new individuals

% take a new set of images of the same individuals to test. 


test_starting_individual=6;%maximum value = 40
test_ending_individual=10;
test_number_of_individuals=test_ending_individual-test_starting_individual+1;%maximum value = 400

test_starting_images_per_individual=6;%maximum value = 10
test_ending_images_per_individual=10;%maximum value = 10
test_number_of_images_per_individual=test_ending_images_per_individual-test_starting_images_per_individual+1;

test_number_total_images=test_number_of_individuals*test_number_of_images_per_individual;
test=zeros(4096,test_number_total_images);

imagecount=1;
for i=test_starting_individual:test_ending_individual
    for j=test_starting_images_per_individual:test_ending_images_per_individual
        test(:,imagecount)=faces(:,(i-1)*10+j);
        imagecount=imagecount+1
    end
end

reconPureTestNew=zeros(4096,test_number_total_images);
reconDcapTestNew=zeros(4096,test_number_total_images);
rmsetestNew=zeros(test_number_total_images,1);

for imagecount=1:test_number_total_images

%% Pure reconstruction

% imagecount=testvalue;

InputImage=test(:,imagecount);
temp=InputImage;
% m=mean(temp);
% st=std(temp);
% temp=(temp-m)*80/st+100;
NormImage = temp;
% Difference = temp-m;
    
ProjectionPureTestNew=zeros(test_number_total_images,1);    
for i=1:test_number_total_images
    ProjectionPureTestNew(i,1)=dot(Eimage(:,i),NormImage-avg);
end

% Projection=Projection/norm(Projection);
% norm(Projection)
reconPureTestNew(:,imagecount)=Eimage*ProjectionPureTestNew+avg;

    figure(1);
    subplot(1,3,1);
    Image=reshape(NormImage,64,64);
    image(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Original Image');
    title(str,'FontSize',12);

    
    subplot(1,3,2);
    Image=reshape(reconPureTestNew(:,imagecount),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Reconstructed image \n using all eigen vectors');
    title(str,'FontSize',12);
    

%% Reconstruction with dcap

% dcap=24;
% dcap= 25;

BasisEVector=Eimage(:,1:dcap);
ProjectionDcapTestNew=zeros(dcap,1);

for i=1:dcap
    ProjectionDcapTestNew(i,1)=dot(BasisEVector(:,i),NormImage-avg);
end

% Projection=Projection/norm(Projection);
% norm(Projection)
reconDcapTestNew(:,imagecount)=BasisEVector*ProjectionDcapTestNew+avg;

total=0;
for j=1:4096
    total=sum(reconDcapTestNew(j,imagecount)-test(j,imagecount)).^2;
end
rmsetestNew(imagecount,1)=sqrt(total/4096);

    
    subplot(1,3,3);
    Image=reshape(reconDcapTestNew(:,imagecount),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    pause(0.2);
    str=sprintf('Reconstructed image.\n Dcap = %d\n RMSE=%f',dcap,rmsetestNew(imagecount,1));
    title(str,'FontSize',12);
    
    
    str=sprintf('Report PCA/ReconstructionImageOfDiffIndividuals%d.jpg',imagecount);
    saveas(gcf,str);
    

end


%% Testing for different images(not faces)

number_tests=input('Enter the number of tests you want to do\n');
rmsetestNewDiff=zeros(number_tests,1);

for imagecount=1:number_tests
[filename1, pathname1] = uigetfile({'*.*'},'Get Database');
testimage=double(rgb2gray(imread(filename1)));

InputImage=testimage;
temp=InputImage;
% m=mean(temp);
% st=std(temp);
% temp=(temp-m)*80/st+100;
NormImage = reshape(temp,64*64,1);
% Difference = temp-m;
    
ProjectionPureTestNew=zeros(test_number_total_images,1);    
for i=1:test_number_total_images
    ProjectionPureTestNew(i,1)=dot(Eimage(:,i),NormImage-avg);
end

% Projection=Projection/norm(Projection);
% norm(Projection)
reconPureTestNew(:,imagecount)=Eimage*ProjectionPureTestNew+avg;

    figure(1);
    subplot(1,3,1);
    Image=reshape(NormImage,64,64);
    image(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Original Image');
    title(str,'FontSize',12);

    
    subplot(1,3,2);
    Image=reshape(reconPureTestNew(:,imagecount),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    str=sprintf('Reconstructed image \n using all eigen vectors');
    title(str,'FontSize',12);
    

%% Reconstruction with dcap

% dcap=24;
% dcap= 25;

BasisEVector=Eimage(:,1:dcap);
ProjectionDcapTestNew=zeros(dcap,1);

for i=1:dcap
    ProjectionDcapTestNew(i,1)=dot(BasisEVector(:,i),NormImage-avg);
end

% Projection=Projection/norm(Projection);
% norm(Projection)
reconDcapTestNew(:,imagecount)=BasisEVector*ProjectionDcapTestNew+avg;

total=0;
for j=1:4096
    total=sum(reconDcapTestNew(j,imagecount)-test(j,imagecount)).^2;
end
rmsetestNewDiff(imagecount,1)=sqrt(total/4096);

    subplot(1,3,3);
    Image=reshape(reconDcapTestNew(:,imagecount),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    pause(0.2);
    str=sprintf('Reconstructed image.\n Dcap = %d\n RMSE=%f',dcap,rmsetestNewDiff(imagecount));
    title(str,'FontSize',12);
    
    str=sprintf('Report PCA/ReconstructionImageOfNonFace%d.jpg',imagecount);
    saveas(gcf,str);


end



%% Nearest Neighbour test

number_tests=input('Enter the number of tests you want to\n do for testing the nearest neighbour face recognition scheme\n');
for i=1:number_tests
num=input('Enter the face you want to recognize');

InputImage=faces(:,num);
NormImage = InputImage;
    
BasisEVector=Eimage(:,1:dcap);
ProjectionNN=zeros(dcap,1);

for i=1:dcap
    ProjectionNN(i,1)=dot(BasisEVector(:,i),NormImage-avg);
end

for i=1:number_total_images
    
    Difference(:,i)=WeightSet(:,i)-ProjectionNN;
    Difference(:,i)=Difference(:,i).^2;
    DiffNorm(i)=norm(Difference(:,i));
    
end

[MinDiff Ind]=min(DiffNorm);
Index=Ind*2-1;

    subplot(1,2,1);
    Image=reshape(InputImage,64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    pause(0.2);
    str=sprintf('Tested Image');
    title(str,'FontSize',12);
    
    subplot(1,2,2);
    Image=reshape(faces(:,Index),64,64);
    imagesc(Image)
    colormap(gray(256));
    daspect([1 1 1]);
    pause(0.2);
    str=sprintf('Corresponding Individual number%d',uint8(Index/10+1));
    title(str,'FontSize',12);
    
    str=sprintf('Report PCA/NearestNeighbourPrediction%d.jpg',Index);
    saveas(gcf,str);


end
    

