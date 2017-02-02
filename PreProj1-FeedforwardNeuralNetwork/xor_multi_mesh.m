file1 = 'testing_data_hill.csv';
file2 = 'testing_data_valley.csv';
data1 = csvread(file1);
data2 = csvread(file2);
x1 = data1(:,1);
x2 = data2(:,1);
y1 = data1(:,2);
y2 = data2(:,2);
z1 = data1(:,3);
z2 = data2(:,3);


%%% Hill Mesh %%%
tri1 = delaunay(x1,y1);
trimesh(tri1,x1,y1,z1, 'FaceAlpha',.67,...
    'EdgeColor','interp',...
    'FaceColor','interp')
hold on
plot3(x1,y1,z1,'.','MarkerSize',15)

colormap hsv
axis tight
material shiny
camlight headlight


%%% Valley Mesh %%%
figure 
tri2 = delaunay(x2,y2);
trimesh(tri2,x2,y2,z2, 'FaceAlpha',.67,...
    'EdgeColor','interp',...
    'FaceColor','interp')
hold on
plot3(x2,y2,z2,'.','MarkerSize',15)

colormap hsv
axis tight
material shiny
camlight headlight