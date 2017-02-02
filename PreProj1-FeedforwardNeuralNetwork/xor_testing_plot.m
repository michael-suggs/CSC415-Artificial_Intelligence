filename = 'testing_data.csv';
data = csvread(filename);
x = data(:,1);
y = data(:,2);
z = data(:,3);


%%% Mesh Plot %%%
tri = delaunay(x,y);
trimesh(tri,x,y,z, 'FaceAlpha',.67,...
    'EdgeColor','interp',...
    'FaceColor','interp')
hold on
plot3(x,y,z,'.','MarkerSize',15)

colormap hsv
axis tight
material shiny
camlight headlight


%%% Surface Plot %%%
figure
trisurf(tri,x,y,z,'FaceColor','interp',...
    'EdgeColor','interp',...
    'LineWidth',1,...
    'FaceLighting','gouraud')
hold on
stem3(x,y,z,'Marker','h',...
    'MarkerFaceColor','k',...
    'MarkerSize',3)


colormap hsv
axis tight
material shiny
camlight headlight