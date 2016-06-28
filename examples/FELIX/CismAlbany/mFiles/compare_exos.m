
fdir_in = '../';
exo_fname_in = 'greenland_standalone-albanyT.exo';
exo_fname2_in = 'inputFiles/greenland_cism-albanyT.exo';

nLevels=11;  %careful! this needs to be compatible with the grids

isAlbanyMeshOrderColumnWise = true;

%% import fields of exo mesh

s_exo_names = struct('x', 'coordx','y', 'coordy', 'z', 'coordz', 'basal_friction','basal_friction', 'flow_factor' ,'flow_factor',  'Velx', 'solution_x', 'Vely', 'solution_y', 'sh', 'surface_height', 'thk', 'thickness', 'temperature', 'temperature');

s_geo = exo_read( [fdir_in, exo_fname_in], s_exo_names);
s_geo2 = exo_read( [fdir_in, exo_fname2_in], s_exo_names);

%% extract fields from s_geo struct

exo_beta = s_geo.basal_friction(:,end);
exo2_beta = s_geo2.basal_friction(:,end);
exo_Velx = s_geo.Velx(:,end);
exo_Vely = s_geo.Vely(:,end);
exo2_Velx = s_geo2.Velx(:,end);
exo2_Vely = s_geo2.Vely(:,end);
exo_sh = s_geo.sh(:,end);
exo2_sh = s_geo2.sh(:,end);
exo_thk = s_geo.thk(:,end);
exo2_thk = s_geo2.thk(:,end);
if(~isempty(s_geo.flow_factor))
  exo_flowfactor = s_geo.flow_factor(:,end);
end
exo2_flowfactor = s_geo2.flow_factor(:,end);
exo_temperature= s_geo.temperature(:,end);
exo2_temperature = s_geo2.temperature(:,end);
coords1 = [s_geo.x,s_geo.y, s_geo.z];
coords2 = [s_geo2.x,s_geo2.y, s_geo2.z];

size3d = length(exo_beta);
size2d = size3d/nLevels;
size3d_cell = length(exo_temperature);
size2d_cell = size3d_cell/(nLevels-1);

albany2cism_node_map = ones(nLevels,1)*(1:size2d) + size2d*(nLevels-1:-1:0)' *ones(1,size2d);
albany2cism_elem_map = ones(nLevels-1,1)*(1:size2d_cell) + size2d_cell*(nLevels-2:-1:0)' *ones(1,size2d_cell);

if(~isAlbanyMeshOrderColumnWise)
  I2 = reshape(albany2cism_node_map',size3d,1);
  I2_cell = reshape(albany2cism_elem_map',size3d_cell,1);
else
  I2 = reshape(albany2cism_node_map,size3d,1);
  I2_cell = reshape(albany2cism_elem_map,size3d_cell,1);
end

disp(['coordinates mismatch [km]: ',  num2str(norm(coords1(:,:) - coords2(I2,:), inf))]);

disp(['thickness mismatch [km]: ',  num2str(norm(exo_thk(:,1) - exo2_thk(I2,1), inf))]);

disp(['surface heigth mismatch [km]: ',  num2str(norm(exo_sh(:,1) - exo2_sh(I2,1), inf))]);

disp(['basal_friction mismatch [kPa m/yr]: ',  num2str(norm(exo_beta(:,1) - exo2_beta(I2,1), inf))]);

disp(['x comp of velocity mismatch [m/yr]: ',  num2str(norm(exo_Velx(:,1) - exo2_Velx(I2,1), inf))]);

disp(['y comp of velocity mismatch [m/yr]: ',  num2str(norm(exo_Vely(:,1) - exo2_Vely(I2,1), inf))]);

if(~isempty(s_geo.flow_factor))
  disp(['flow factor mismatch: ',  num2str(norm((exo_flowfactor(:,1) - exo2_flowfactor(I2_cell,1)), inf))]);
end

disp(['temperature mismatch [K]: ',  num2str(norm((exo_temperature(:,1) - exo2_temperature(I2_cell,1)),inf))]);

%% print variables
% coords2d = coords1(1:size2d,1:2);
% dx = 16; dy=16;
% x=min(coords2d(:,1)):dx:max(coords2d(:,1));
% y=min(coords2d(:,2)):dy:max(coords2d(:,2));
% [X,Y] = meshgrid(x,y);
% coords2d_grid = [reshape(X,numel(X),1), reshape(Y, numel(Y),1)];
% [z, i1,i2] =  intersect(coords2d, coords2d_grid, 'rows');
% z = exo_Velx(:,1) - exo2_Velx(I2,1);
% z = z((1:size2d)+0*(nLevels-1)*size2d);
% Z = zeros(size(coords2d_grid,1),1);
% Z(i2) = z(i1);%(1:nLevels:end);
% figure(2); pcolor(X,Y,reshape(Z, size(X,1), size(X,2))); colorbar; shading interp
