function [] = couple_SOCS2PISM(trg_yr)
%couple_SOCS2PISM
%         crops a temperature field from the most recent SOCS output
%trg_yr=2011;
%% set path and file names
[~,pre_path]=set_machine_path('027');
%pre_path = realpath(pre_path);
trg_path = realpath(read_PISM2SOCS_input('INI_PATH'));

src_cse = read_PISM2SOCS_input('SOCS_SRC_CSE');
src_cse = ['avg_' src_cse];
src_pth = realpath(read_PISM2SOCS_input('SOCS_SRC_OUT_PTH'));
%src_fle = realpath(read_PISM2SOCS_input('ROMS_SRC_FLE'));

src_var = read_PISM2SOCS_input('S2P_VAR_SRC');
trg_var = read_PISM2SOCS_input('S2P_VAR_TRG');

% src_tst is replaced with automatic value
%src_tst = maxmax(read_PISM2SOCS_input('ROMS_SRC_TSP'));

s2p_cpl_ini = read_PISM2SOCS_input('S2P_CPL_INI');
zro_fle = [trg_path '/' s2p_cpl_ini];
s2p_cpl_cse = read_PISM2SOCS_input('S2P_CPL_CSE');

trg_fle = [trg_path '/' s2p_cpl_cse '_' num2str(trg_yr) '.nc'];

%% retrieve latest SOCS outputs
disp(['-- couple_SOCS2PISM: looking for case ' src_cse])
dat_lst = data_case_V05(src_pth,'ocean_time',src_var,src_cse);

%% calculate nc records to be retrieved from average file and interval weights
[~,b]=calendar_index(dat_lst.src_tme,dat_lst.src_dpyr,dat_lst.src_ref_dte);
% assuming time stamps of avg records are centred on the interval
c = find(b.y>trg_yr,1,'first');
d = find(b.y<(trg_yr+1),1,'last');
nc_n=c:d;
int_w = ones(size(nc_n))';
% create bounds of the intervals
% d = find(b.y>(trg_yr+1),1,'first');
% if isempty(d)
%     d=length(b.y);
% end
% 
% c = find(b.y<trg_yr,1,'last');
% if isempty(c)
%     c=1;
%     nc_n=c:d;
%     int_w=diff(b.y(c:d));
%     artificially add first interval
%     int_w=[mean(int_w); int_w];
%     int_w(1)=int_w(1)-max([trg_yr-(b.y(c)-int_w(1)), 0]);
% else
%     nc_n=(c+1):d;
%     int_w=diff(b.y(c:d));
%     see how much weight first and last interval receive
%     int_w is implicitly normalized to 1 (b.y is the floating point year index)
%     int_w(1)=int_w(1)-max(trg_yr-b.y(c),0);
% end
% int_w(end)=int_w(end)-max([b.y(d)-(trg_yr+1), 0]);

int_w = permute(int_w,[4 3 2 1]);

%% retrieve grid file of the latest SOCS output
src_grd_fle = [char(dat_lst.src_grd_pth(dat_lst.fle_ind(nc_n(end)))) char(dat_lst.src_grd_fle(dat_lst.fle_ind(nc_n(end))))];
%src_grd_fle = [pre_path dat_lst.src_grd_fle];
%src_grd_cse = dat_lst.src_grd_fle(9:end-3);
src_grd_cse = src_grd_fle(strfind(src_grd_fle,'_grd_')+5:strfind(src_grd_fle,'.nc')-1);

%% copy file template to new PISM forcing file
copyfile(zro_fle,trg_fle)

%% retrieve target coordinates
trg_lat = ncread(trg_fle,'lat');
trg_lon = ncread(trg_fle,'lon');

trg_lon(trg_lon>180)=trg_lon(trg_lon>180)-360;
[x84_trg,y84_trg]=polarstereo_fwd(trg_lat,trg_lon,earth_rad('ROMS')*1000,0.08181919,-71,0);

[crp_ind]=crop_index(src_grd_fle,'lat_rho','lon_rho',trg_lat,trg_lon);

%% get source field & coordinates
% retrieve sigma layer depths
nc_info = ncinfo(src_grd_fle);
nc_vnme = {nc_info.Variables.Name};
if contains(nc_vnme,'z_w')
    z_w  = ncread(src_grd_fle,'z_w',[crp_ind(3) crp_ind(1) 1],[diff(crp_ind(3:4))+1 diff(crp_ind(1:2))+1 Inf]);
else
    load([trg_path '/pre_data/mapping_' src_grd_cse '.mat'],'trg_grd')
    z_w = permute(trg_grd.z_w(:,crp_ind(1):crp_ind(2),crp_ind(3):crp_ind(4)),[3 2 1]);
end


% initialize field

fld = zeros([diff(crp_ind(3:4))+1 diff(crp_ind(1:2))+1 size(z_w,3)-1 length(nc_n)]);

for i=1:length(nc_n)
    src_fle = [src_pth '/' char(dat_lst.fle_nmes(dat_lst.fle_ind(nc_n(i))))];
    fld(:,:,:,i) = ncread(src_fle,src_var,[crp_ind(3) crp_ind(1) 1 dat_lst.nct_ind(nc_n(i))],[diff(crp_ind(3:4))+1 diff(crp_ind(1:2))+1 Inf 1]);
end
% weighted average over time dimension
fld = sum(bsxfun(@times,fld,int_w)./sum(int_w),4);
%src_grd_fle = [dat_lst.src_grd_pth '/' dat_lst.src_grd_fle];
src_lat = ncread(src_grd_fle,'lat_rho',[crp_ind(3) crp_ind(1)],[diff(crp_ind(3:4))+1 diff(crp_ind(1:2))+1]);
src_lon = ncread(src_grd_fle,'lon_rho',[crp_ind(3) crp_ind(1)],[diff(crp_ind(3:4))+1 diff(crp_ind(1:2))+1]);

%% calculate depth average
Hz = diff(z_w,1,3);
clear z_w
fld_avg = sum(fld.*Hz,3)./sum(Hz,3);
clear fld Hz

%% create metric coordinates for interpolation
src_lon(src_lon>180)=src_lon(src_lon>180)-360;
[x84_src,y84_src]=polarstereo_fwd(src_lat,src_lon,earth_rad('ROMS')*1000,0.08181919,-71,0);

F = scatteredInterpolant(x84_src(isfinite(fld_avg)),y84_src(isfinite(fld_avg)),fld_avg(isfinite(fld_avg)));
F.Method='natural';
fld = F(x84_trg,y84_trg);
fld = fld+274.15-ncread(zro_fle,trg_var);
ncwrite(trg_fle,trg_var,fld);
disp(trg_fle)

end

