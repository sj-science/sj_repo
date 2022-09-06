function [grd_file]=make_bathy_PISM2SOCS(SRC_YR) 
% SRC_PTH path to PISM output
% SRC_FNM file name of PISM output
% SRC_TSP time step in PRISM output to be processed
% TRG_CSE file name of the new ROMS bathymetry
% TRG_PTH path to new ROMS bathymetry file
% MRG_CSE file name of the parent ROMS bathymetry
% MRG_PTH path to parent ROMS bathymetry file

warning('off', 'all')
% test parameters - comment out when running as function -----------------------
% 
% SRC_PTH = ['~/03_science/05_projects/14_SeaRise/PISM_docs_n_data/'];
% SRC_FNM = ['Nick-T2-120ka.nc'];
% SRC_TSP = 1;
% TRG_CSE = ['PISMt1_2_SOCStn'];
% TRG_PTH = ['~/08_data/027/preparation/'];
% MRG_CSE = ['SOCS_V21'];
% MRG_PTH = ['~/08_data/027/input/'];
%-------------------
SRC_PTH = realpath([read_PISM2SOCS_input('PISM_SRC_PTH')]);
SRC_FNM = [read_PISM2SOCS_input('PISM_SRC_FNM')];
%SRC_YR = max(read_PISM2SOCS_input('PISM_SRC_YR'));
%TRG_CSE = [read_PISM2SOCS_input('ROMS_TRG_GRD_CSE')];
%TRG_PTH = realpath([read_PISM2SOCS_input('ROMS_TRG_GRD_PTH')]);
MRG_CSE = [read_PISM2SOCS_input('ROMS_TRG_MRG_CSE')];
MRG_PTH = realpath([read_PISM2SOCS_input('ROMS_TRG_MRG_PTH')]);
% ------------------------------------------------------------------------------
% check inputs
if ~strcmpi(SRC_PTH(end),'/')
   SRC_PTH = [SRC_PTH '/'];
end
%if ~strcmpi(TRG_PTH(end),'/')
%   TRG_PTH = [TRG_PTH '/'];
%end
if ~strcmpi(MRG_PTH(end),'/')
   MRG_PTH = [MRG_PTH '/'];
end
if ischar(SRC_YR)
    SRC_YR = str2double(SRC_YR);
end
% ------------------------------------------------------------------------------

% grid configuration
N_grd=17;
% start processing from
proc_stage=1; 
%1: interpolation from rawspon
%2: mask definition

%% script execution     
N_sav=1; % temp storage, to restart script from proc_stage
save_file  = 'on';
plt_sig_layers = 'off';
mask_control_plots=0;
pick_spots = 'off'; % no data are written to nc file, instead a plot to pick spots is generated

global S B M ARVD
% don't use S B M for anything else
%% reading execution parameter
[S, B, M, ARVD ] = vertical_grid_config(N_grd);

sim_numbr =     S.sim_numbr;
sim_case  =     S.sim_case;
grd_case  =     S.grd_case; 
mrg_case  =     S.mrg_case;
bathymetry =    B.bathymetry;
limit_depth =   B.limit_depth;
circumpolar_model=B.circumpolar_model;
%source =        B.source;
avg_filter =    B.avg_filter;
n_sponge =      B.n_sponge;
sponge_layer =  B.sponge_layer;
sponge_dir  =   B.sponge_dir;
%pre_smooth =    B.pre_smooth;
%smooth_rx1 =    B.smooth_rx1;
%smooth_rx0 =    B.smooth_rx0;
%smooth_funnel = B.smooth_funnel;
Wc_min =        B.Wc_min;
shave_n_dig_IS= B.shave_n_dig_IS;
rx0 =           B.rx0;
rx1_h =         B.rx1_h;
rx1_i =         B.rx1_i;
ice_shelf =     M.ice_shelf;
kill_ice =      M.kill_ice;
ice2land =      M.ice2land;
icsh2ocean =    M.icsh2ocean;
shlf_sze =      M.shlf_sze;
%max_ice_cliff = M.max_ice_cliff;
max_iterations =M.max_iterations;
fill_narrows =  M.fill_narrows;
fill_narrows_sub=M.fill_narrows_sub;
nuke_islands =  M.nuke_islands;
%isl_sze =       M.isl_sze;
margin_width =  M.margin_width;
M.mask_control_plots=mask_control_plots;

%% entries that will be overwritten with input from function
%sim_case  = TRG_CSE;
mrg_case = MRG_CSE;

%% end overwriting -----------------------------------------
%% path and in/out file definition

[roo_path, pre_path, inp_path, out_path, pro_path, tmp_path] = set_machine_path(sim_numbr,4);
pre_path = [realpath(pre_path) '/'];
pro_path = [realpath(pro_path) '/'];
grd_file = [pre_path sim_numbr '_grd_' grd_case '.nc']; % netcdf source file

sim_file = [pre_path sim_numbr '_grd_' sim_case '.nc']; % netcdf output file
%sim_file = [TRG_PTH sim_numbr '_grd_' sim_case '.nc']; % netcdf output file

map1_file = [pre_path 'pre_data/mapping_' sim_case '.mat']; % mapping file
map2_file = [pro_path 'pro_data/mapping_' sim_case '.mat']; % mapping file

strg_file = [pre_path 'pre_data/fill_bathy' num2str(N_sav) '_tmp_delete'  '.mat' ];
% if mrg_case~=0
%     [ mrg_sim_numbr, ~,~, mrg_files, ~, ~, ~, opt_out_path] = nc_output_case( mrg_case);
%     mrg_file = [opt_out_path C2M(mrg_files(1))];
% end
%% save script
if strcmpi(save_file, 'on')
 %   g = save_script([realpath(read_PISM2SOCS_input('LOG_FLE_PTH')) '/fill_bathy_script_' sim_numbr '_' sim_case],mfilename);
end


%% copy grd file with matrix to new grd file to be filled
if strcmpi(save_file, 'on')&&~exist(sim_file,'file')
    disp(['copy ' grd_file])
    disp(['to become'])
    disp([sim_file])
    copyfile(grd_file,sim_file);
    
    grd_file = sim_file;
    clear sim_file
    new_grd_fle=1;
else
    new_grd_fle=0;
    grd_file = sim_file;
    clear sim_file
end
%% retrieve information on the grd file
nc_info = ncinfo(grd_file);
nc_vnme = {nc_info.Variables.Name};

%% establish target coordinates

LON_TRG = ncread(grd_file,'lon_rho');
LAT_TRG = ncread(grd_file,'lat_rho');

%% a few settings that depend on the size of the target grid
% size parameter
[N_lon,N_lat] = size(LON_TRG);

if exist('pro_bnd','var') && strcmpi(pro_bnd,'all')
    pro_bnd = [1 N_lon 1 N_lat];
end
MSK_his = NaN([N_lon N_lat 4 1]); % storage for 1:lmask 2:imask 3:rmask 4:omask
%% ========================================================================
his_cnt = 1; % history counter for storing fields
timer0 = tic;
if proc_stage==1
    %% 1a) load topo data, knitting, define masks
    
    
    disp('== proc stage 1 =================================================')
    disp(['Loading, cropping, blending, interpolating of H and I ...'])
    
    %[H, I,data_source] = retrieve_src_bathy_surfaces(source,LAT_TRG,LON_TRG,ice_shelf,circumpolar_model);
    
    data_source = 'H & I: PISM model';
    
    % yields H and I interpolated onto target grid
    [H,I] = get_topoPISM(LAT_TRG,LON_TRG,SRC_PTH,SRC_FNM,SRC_YR );
    H(isnan(H))=4000;
    % this is to fill the surrounding of Antarctica with deep ocean
    % floor. this will be replaced with other topography later in
    % the process
    I(isnan(I))=0;
    
    % check for NaN in interpolated data
    if check4NaN(H,I,['1a) interpolating fields. STOP'])==1
        return
    else
        disp(['1a) No NaNs in the interpolated fields.'])
    end
    
    
    disp(['   took ' num2str(round_dec(toc(timer0),1)) 's'])
    
     
    %% 1b) define lmask / imask / omask (no bathy manipulation)
    % land mask: 1:land 0:ice/open ocean
    % - lmask determines the potential computation domain
    %lmask = H<Wc_min/2;
    lmask = H<=0|isnan(H);
  
    % ice  mask: 1:ice 0:open ocean/land
    % - imask is ice shelf where melting happens and ice sheet where potential
    %   future melting can happen
    imask = I~=0;
    imask(lmask==1)=0;
    
    % ocean mask: 1:open ocean 0:ice/land
    % open ocean where is no land and no ice -> open ocean
    omask = (lmask==0&imask==0);
    
   
    %% 1c) establish rmask and check consistency of all masks
    
    % rho mask: 1:ocean 0:grounded ice/land
    % - rmask sets the current computation domain
    % - first step rmask: where no land -> rmask==1
    rmask = (lmask==0);
    % set water column on unprocessed bathy surfaces
    Wc=H+I;
    % - second step rmask: rmask==0 where the (unprocessed) water colum 
    %   under ice is less than half minimum Wc
    if shave_n_dig_IS==1 % if shallower than 50% min wc make grounded
        rmask(imask==1&Wc<Wc_min/2)=0;
    elseif shave_n_dig_IS==2 % if shallower than 10% min wc make grounded
        rmask(imask==1&Wc<Wc_min/10)=0;
    else % if shallower than min wc make grounded
        rmask(imask==1&Wc<Wc_min)=0;
    end

    %% 1d) create check mask that should be
    % 1:land | 2:open ocean | 4: grounded ice sheet | 6: ice shelf
    if check_masks(lmask,rmask,imask,0,'Initialization')==1
        return
    else
        % update ocean mask
        omask = (lmask==0&imask==0);
        if mask_control_plots==1
            plot_masks(lmask,rmask,imask,('1d initialization'));
        end
    end
    
    %% 1e) check for conventions: 1:  H>0 (ocean) => I<=0
    %                             2:  H<=0 (land) => I>=0
    if any(H(:)>=0&I(:)>0)
        disp(['- inconsistent topography - ice draft above sea level over ocean.'])
        disp(['- set ' num2str(sumsum(H(:)>=0&I(:)>0)) ' spots to -1m ice draft'])
        %[~,tmp] = check_masks(lmask,rmask,imask,1);
        %tmp(H>=0&I>0)=10;
        %figure; imagesc(tmp); title(['ice draft above sea level over ocean ' num2str(sumsum(H(:)>=0&I(:)>0))])
        I(H>=0&I>0)=-1;
    else
        disp(['- consistent topography: H>0 (ocean) => I<=0 '])

    end
    if any(H(:)<0&I(:)<0)
        %figure; imagesc(H<0&I<0); title('ice draft below sea level over land')
        disp(['- inconsistent topography - ice draft below sea level over land '])
        disp(['- set ' num2str(sumsum(H(:)<0&I(:)<0)) ' ice spots to bed rock value'])
        %[~,tmp] = check_masks(lmask,rmask,imask,1);
        %tmp(H<0&I<0)=10;
        %figure; imagesc(tmp); title(['ice draft below sea level over land ' num2str(sumsum(H(:)<0&I(:)<0))])
        I(H<0&I<0)=-H(H<0&I<0);
    else
        disp(['- consistent topography: H<=0 (land) => I>=0'])
    end
    
    if check_all(H,I,lmask,rmask,imask,' after mask initialization.')~=0
        return
    end
%    [rkey,cmask]=check_all(H,I,lmask,rmask,imask,' after mask initialization.');
   

    if ~any(contains(nc_vnme,'zice_orig'))
        nccreate(grd_file,'zice_orig','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','single');
        ncwriteatt(grd_file,'zice_orig','long_name','original ice draft, interpolated');
        ncwrite(grd_file,'zice_orig',I)
    end
    if ~any(contains(nc_vnme,'h_orig'))
        nccreate(grd_file,'h_orig','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','single');
        ncwriteatt(grd_file,'h_orig','long_name','original bathymetry, interpolated');
        ncwrite(grd_file,'h_orig',H)
    end
 
    
    %% save vars to save time on iterations to come
    % store initial H & I -------------------------------------------------
    H_his(:,:,his_cnt) = H;  % stores initial bathymetry after interpolation [H is positive]
    I_his(:,:,his_cnt) = I;  % stores initial bathymetry after interpolation [I is negative]
    
    MSK_his(:,:,1:4,his_cnt) = cat(3,lmask,imask,rmask,omask);
    save(strg_file,'H_his','I_his','MSK_his','data_source')
    disp(['Saving masks and fields after bathy interpolation and mask ini ' num2str(his_cnt)])
elseif proc_stage==2
    disp(['Loading fields after bathy interpolation and mask initialization.'])
    load(strg_file)
    H = H_his(:,:,his_cnt);
    I = I_his(:,:,his_cnt);
    lmask = MSK_his(:,:,1,his_cnt);
    imask = MSK_his(:,:,2,his_cnt);
    rmask = MSK_his(:,:,3,his_cnt);
    omask = MSK_his(:,:,4,his_cnt);
end

%% ========================================================================
%% 2) treat masks

his_cnt = his_cnt+1;
if proc_stage<3
    if check_all(H,I,lmask,rmask,imask,' 2) before treating masks.')~=0
        return
    end

    if proc_stage>1 && mask_control_plots==1
        plot_masks(lmask,rmask,imask,('2) before treating masks'));
    end
    
    
    %% 2a) lmask: nuke islands    
    if strcmpi(nuke_islands,'on')
        disp(' ')
        disp('== proc stage 2a ================================================')
        disp('Removing small islands around the edge of the domain (away from ice shelves)')
        
        % this is the only time where land points are release to become ocean
        % mainly to remove small islands near the boundaries
        [H,lmask,rmask]=remove_islands(H,lmask,rmask,margin_width);
        
        %% check mask consistency
        if mask_control_plots==1
            plot_masks(lmask,rmask,imask,('2a) after nuking rmask islands'));
        end
        
        if check_all(H,I,lmask,rmask,imask,'2a) after nuking rmask islands')~=0
            return
        end

    end
    
    %% 2b-d : lmask remove of narrows | rmask remove narrows | remove small IS | gaps between IS and land

    N_jj=1;
    ID_oce2ice = [];
   
    for jj=1:N_jj
        
        %% 2b) lmask: remove narrow straits and holes         
        disp(' ')
        disp('== proc stage 2b ================================================')
      
        % whole domain
        disp(['Operating on whole domain, removing narrows of ' num2str(fill_narrows) ' in lmask.'])
      
        [lmask,rmask,imask,tmp_msk]=remove_lmask_narrows(lmask,rmask,imask);
        
        %% check mask consistency
        if mask_control_plots==1 %&& jj==1
            plot_masks(lmask,rmask,imask,('2b) after removing lmask narrows.'));
            %figure;imagesc(tmp_msk);title('del_mask')
        end
        if check_ice_mask_consistency(I,imask,['after stage 2b run ' num2str(jj)])==1 || ...
                check_masks(lmask,rmask,imask,0,'lmask full region: fill narrows & holes')==1
            return
        end

        % specific region - needs update
        if ~isempty(fill_narrows_sub)
            for i_sub=1:size(fill_narrows_sub,1)
                fill_n = C2M(fill_narrows_sub(i_sub,1));
                sub_i = C2M(fill_narrows_sub(i_sub,2));
                sub_j = C2M(fill_narrows_sub(i_sub,3));
               % brdr_flag = C2M(fill_narrows_sub(i_sub,5));
                disp(' ')
                disp('== proc stage 2b1 ===============================================')
                disp(['Operating on sub domain ' C2M(fill_narrows_sub(i_sub,4)) ' , removing narrows of ' num2str(fill_n) '.'])
                
                tmp_lmask = lmask(sub_i(1):sub_i(2),sub_j(1):sub_j(2));
                tmp_rmask = rmask(sub_i(1):sub_i(2),sub_j(1):sub_j(2));
                tmp_imask = imask(sub_i(1):sub_i(2),sub_j(1):sub_j(2));
                
                [tmp_lmask,tmp_rmask,tmp_imask]=remove_lmask_narrows(tmp_lmask,tmp_rmask,tmp_imask,fill_n);
               
                lmask(sub_i(1):sub_i(2),sub_j(1):sub_j(2)) = tmp_lmask;
                imask(sub_i(1):sub_i(2),sub_j(1):sub_j(2)) = tmp_imask;
                rmask(sub_i(1):sub_i(2),sub_j(1):sub_j(2)) = tmp_rmask;
                clear tmp_lmask tmp_imask tmp_rmask
            end
            
            %% check mask consistency
            if mask_control_plots==1 && jj==1
                plot_masks(lmask,rmask,imask,('2b1) after removing lmask narrows / sub region'));
              
            end
            if check_ice_mask_consistency(I,imask,['after stage 2b1. ' ])==1 || ...
                    check_masks(lmask,rmask,imask,0,'lmask sub region: fill narrows & holes')==1
                return
            end
        end
        
        %% 2c) rmask: remove narrow straits and holes
        % a hole on rmask must be a lake under ice. If it is not, you are forced to check!
        disp(' ')
        disp('== proc stage 2c ================================================')
        
        tmp = size(ID_oce2ice,1);
        [ lmask,rmask,imask,omask,ID_oce2ice ] = remove_rmask_narrows_A( lmask,rmask,imask,omask,ID_oce2ice );
          disp(['- added ' num2str(size(ID_oce2ice,1)-tmp) ' spots to ID_oce2ice vector. Total: ' num2str(size(ID_oce2ice,1))])
        % temporaily set I==1 in those spots that will be
        % invented in stage 3
        % This serves to keep consistency between imask and I
        % I=1 will be removed before inventing actual ice draft
        if ~isempty(ID_oce2ice)
            for ido=1:size(ID_oce2ice,1)
                I(ID_oce2ice(ido,1),ID_oce2ice(ido,2))=1;
            end
        end
        %% check mask consistency
        if mask_control_plots==1 %&&jj==1
             plot_masks(lmask,rmask,imask,('2c) after removing rmask narrows'));
        end
        if check_ice_mask_consistency(I,imask,['after stage 2c, run ' num2str(jj)])==1 || ...
                check_masks(lmask,rmask,imask,0,'rmask: narrows & holes')==1
            return
        else
            % update ocean mask
            omask = (lmask==0&imask==0);
        end
        
        %% 2d) cancel small ice shelves/sheets (needs update)
        if strcmpi(icsh2ocean, 'on')
            disp(' ')
            disp('== proc stage 2d ================================================')
            
            disp('Remove small ice shelves, operating on imask and I.')
            [shelves, b] = islands(imask==1&rmask==1);
            %     % find all small ice shelves 4km i.e. 4 grid cells and smaller
            b(b(:,2)>shlf_sze,:) = [];
            N_i2o = size(b,1);
            % remove ice shelves of size 1
            
            I(shelves==0&rmask==1) = 0;
            imask(shelves==0&rmask==1) = 0;
            % all other sizes <= shlf_sze
            for i=1:N_i2o
                I(shelves == b(i,1)&rmask==1) = 0;
                imask(shelves == b(i,1)&rmask==1) = 0;
            end
            disp(['Converted ' num2str(N_i2o) ' small ice shelves of size <= ' num2str(shlf_sze) ' to ocean'])
            
            % rmask AND imask are more or less maintained in their original state to
            % ensure the lateral ocean extent and ice shelf/sheet extent in the model
            % are as close to reality as possible
            
            
            %% check mask consistency
            if check_ice_mask_consistency(I,imask,['after stage 2d, run ' num2str(jj)])==1 || ...
                    check_masks(lmask,rmask,imask,0,'imask: small ice shelves')==1
                return;
            else
                % update ocean mask
                omask = (lmask==0&imask==0);
            end
        end
        %% 2e) work on gaps between ice shelf and land   
        disp(' ')
        disp('== proc stage 2e ================================================')
        disp(['Operating on whole domain, removing narrows of ' num2str(fill_narrows) ' between imask and lmask.'])
        
        % narrow gaps between ice shelf and land will be set to land, we dont
        % invent ice
        % change of policy: invent ice is ok. Save in ID_oce2ice for now
        ii=1;
        tmp = imask;
        while ii<M.max_iterations
            disp('---------------------------------------------------------------')
            disp(['Round ' num2str(ii) ' of ' num2str(M.max_iterations) ' iterations, ice2land narrows.'])
            tmp_mask = double(lmask==0);
            tmp_mask(imask==1) = 5; %1 for ocean, 0 for land, 5 for ice

            del_msk = find_narrows(tmp_mask,fill_narrows,3); %find the narrows
            %lmask(del_msk==1)=1; %set to land
            %disp(['Removing ' num2str(sumsum(del_msk==1)) ' narrow spots between ice and land of >=' num2str(fill_narrows)])
            
            disp(['Setting ocean to ice in ' num2str(sumsum(del_msk==1)) ' spots between ice and land narrower <=' num2str(fill_narrows)'])
            if sumsum(del_msk==1)==0
                %all spots already saved in ID vector
                ii=M.max_iterations;
                a = 0;
            else
                imask(del_msk==1)=1; %set to ice
                % temporaily set I==1 in those spots that will be
                % invented in stage 3
                % This serves to keep consistency between imask and I
                % I=1 will be removed before inventing actual ice draft
                
                I(del_msk~=0)=1;
                % find and store indexes of spots so we can invent ice shelf draft
                % later
                [a, b] = find(del_msk~=0);
                ID_oce2ice = [ID_oce2ice; [a b]];
                ii=ii+1;
            end
            disp(['- added ' num2str(length(a)) ' spots to ID_oce2ice vector. Total: ' num2str(size(ID_oce2ice,1))])
        end
        if mask_control_plots==1 && jj==1  
            figure;imagesc(tmp~=imask);
            title('ocean 2 ice conversion on imask after ice2land narrows.')
        end
        %% check mask consistency
        % to check mask 
        if check_ice_mask_consistency(I,imask,['after stage 2e.'])==1 || ...
                check_masks(lmask,rmask,imask,0,'removing paps btw imask & lmask')==1
            return
        else
            % update ocean mask
            omask = (lmask==0&imask==0);
        end
        
        
         %% 2f) find remaining open water holes, most likely between ice shelf and ice sheet
        disp(' ')
        disp('== proc stage 2f ================================================')
        disp(['Operating on whole domain, removing open water holes. Operating on omask.'])
        
        [a,b] = islands(omask);
        b(b(:,3)==0,:)=[]; % remove all shapes that are not open ocean
        [~,d]= max(b(:,2)); % find and remove biggest ocean area from the mask
        b(d,:)=[];
        del_msk = zeros(size(a));
        if numel(b)>0
            for ib=1:size(b,1)
                del_msk(a==b(ib,1))=1; % index open ocean spots on del_msk
            end
        end
        I(del_msk~=0)=1;
        % This serves to keep consistency between imask and I
        % I=1 will be removed before inventing actual ice draft
        imask(del_msk==1)=1; %set to ice
        [a1, b1] = find(del_msk~=0);
        ID_oce2ice = [ID_oce2ice; [a1 b1]];
        disp(['- added ' num2str(length(a)) ' spots to ID_oce2ice vector. Total: ' num2str(size(ID_oce2ice,1))])
        
        
        %% check mask consistency
        % to check mask 
        if check_ice_mask_consistency(I,imask,['after stage 2f.'])==1 || ...
                check_masks(lmask,rmask,imask,0,'removing holes on omask')==1
            return
        else
            % update ocean mask
            omask = (lmask==0&imask==0);
        end
        
        % update imask - no other masks defined yet
        disp('---------------------------------------------------------------')
        disp(['Updating imask after lmask adjustments: ' num2str(sumsum(imask==1&lmask==1))])
        disp('Should be zero!')
        imask(lmask==1)=0;

        if jj==N_jj
            
            disp(' ')
            disp('== stages 2b to 2e carried out multiple times just to make sure =')
            disp('==================================================================')
        end
    end
    % store initial masks -----------------------------------------------------
    H_his(:,:,his_cnt) = H;
    I_his(:,:,his_cnt) = I;
    MSK_his(:,:,1:4,his_cnt) = cat(3,lmask,imask,rmask,omask);
    ID_oce2ice_his(his_cnt) = {ID_oce2ice};
    save(strg_file,'H_his','I_his','MSK_his', 'ID_oce2ice_his','data_source')
    disp(['Saving masks and fields after mask manipulation ' num2str(his_cnt)])
    disp(' ')
elseif proc_stage==3
    disp(['Loading fields after mask manipulation.'])
    load(strg_file)
    H = H_his(:,:,his_cnt);
    I = I_his(:,:,his_cnt);
    ID_oce2ice = C2M(ID_oce2ice_his(his_cnt));
    lmask = MSK_his(:,:,1,his_cnt);
    imask = MSK_his(:,:,2,his_cnt);
    rmask = MSK_his(:,:,3,his_cnt);
    omask = MSK_his(:,:,4,his_cnt);
end

%% ========================================================================
%% 3) manipulate bathymetry and ice shelf

his_cnt = his_cnt+1;
if proc_stage<4
 
    if check_ice_mask_consistency(I,imask,['before stage 3'])==1 || ...
            check_masks(lmask,rmask,imask,0,'before stage 3')==1
        return;
    end
    % check for NaN
    if check4NaN(H,I,'3)')
        return
    end
%       if any(rmask(:)==1&I(:)>0)
%         disp(['3a) positive ice shelf draft - ice draft above sea level over ocean'])
%         return
%     else
%         disp(['3a) ice shelf draft consistent with rmask - ice draft <=0 over ocean'])
%      end
    %% 3a) invent ice draft here / limit water depth (to keep cfl sane)
    disp(' ')
    disp('== proc stage 3 =================================================')
    if numel(ID_oce2ice)>=2
        % removing the place holder where ice draft needs to be invented
        disp(' ')
        disp('== proc stage 3a - inventing ice draft from ID_oce2ice')
        I(I==1)=0;
        if mask_control_plots==1
            I_tmp=I;
        end
        
        i_cnt=1;
        n_round_ID_oce2ice=ID_oce2ice;
        while (~isempty(n_round_ID_oce2ice)&&i_cnt<5)
            [I,n_round_ID_oce2ice]=invent_oce2ice(I,n_round_ID_oce2ice, M.min_ice_draft);
            i_cnt=i_cnt+1;
        end
        if ~isempty(n_round_ID_oce2ice)
            disp([num2str(size(n_round_ID_oce2ice,1)) 'remaining spots were set to ice free'])
            for i_spt=1:size(n_round_ID_oce2ice,1)
                I(n_round_ID_oce2ice(i_spt,1),n_round_ID_oce2ice(i_spt,2)) =0;
                imask(n_round_ID_oce2ice(i_spt,1),n_round_ID_oce2ice(i_spt,2)) =0;
            end
        end
        

        if mask_control_plots==1
            figure('Position',[10 10 500 800])
            subplot = tight_subplot(2,1,[0.03 0.01],[0.01 0.03],[0.01 0.03]);
            axes(subplot(1));
            imagesc(I_tmp-I);colorbar
            title('invented ice shelf draft')
            axes(subplot(2));
            imagesc(I);colorbar
            title('new ice shelf draft, including invented parts')
        end
    end
    if check_all(H,I,lmask,rmask,imask,'3a) after inventing ice shelf spots')~=0
        return
    end
    
    %% insert function to do rx0 and rx1 smoothing here
    %% --------------------------------------------------------------------
    % convention
    % from here grounded ice is no longer carried.
    % because things go wrong at this point a lot we save different stages
    % of the H & I
%      if any(rmask(:)==1&I(:)>0)
%         disp(['3b positive ice shelf draft - ice draft above sea level over ocean'])
%         return
%     else
%         disp(['3b ice shelf draft consistent with rmask - ice draft <=0 over ocean'])
%      end
    

    H1=H;
    I1=I;
    
    proc_subdomain=1;
    reg_buff = 5;
    crp_ind = [343-reg_buff,947+reg_buff,492-reg_buff,1130+reg_buff];
    %crp_ind = [645,667,980,1013];
    disp(' ')
    if proc_subdomain==1
        disp('== proc stage 3b - processing zice and bathy on sub domain')
        
        H2 = H1(crp_ind(1):crp_ind(2),crp_ind(3):crp_ind(4));
        I2 = I1(crp_ind(1):crp_ind(2),crp_ind(3):crp_ind(4));
        % masks are not changed in this step. They are cropped 
        % for processing I & H only
        imsk2 = imask(crp_ind(1):crp_ind(2),crp_ind(3):crp_ind(4));
        rmsk2 = rmask(crp_ind(1):crp_ind(2),crp_ind(3):crp_ind(4));
        lmsk2 = lmask(crp_ind(1):crp_ind(2),crp_ind(3):crp_ind(4));
        
    else
        disp('== proc stage 3b - processing zice and bathy')
        H2 = H1;
        I2 = I1;
        imsk2 = imask;
        rmsk2 = rmask;
        lmsk2 = lmask;
    end
    if check_ice_mask_consistency(I2,imsk2,['before bathy & draft processing (optional on subdomain)]'])==1 || ...
            check_masks(lmsk2,rmsk2,imsk2,0,'Before bathy & draft processing (optional on subdomain)')==1
        return
    end
     
    % simplify masks for bathy processing i.e. carry no ice sheet
    imsk_simp = imsk2;
    rmsk_simp = rmsk2;
    lmsk_simp = lmsk2;
    imsk_simp(rmsk_simp==0)=0;
    lmsk_simp(rmsk_simp==0)=1;
    
    I_simp=I2;
    H_simp=H2;
    I_simp(rmsk_simp==0)=0;
    H_simp(rmsk_simp==0)=NaN;
    
    if check_all(H,I,lmask,rmask,imask,'3b) after simplifying masks')~=0
        return
     end
%     if check_ice_mask_consistency(I_simp,imsk_simp,['after simplifying masks'])==1 || ...
%             check_masks(lmsk_simp,rmsk_simp,imsk_simp,0,'After simplifying masks')==1
%         return
%     end
      %% temporary disable true function to fix lp_solve
    [H4,I4]=ROMS_rx0_rx1_processing(H_simp,double(rmsk_simp),I_simp,double(imsk_simp),strg_file,[0 16 9 16 8 6 5 14 16 14 2]); 
%    [H4,I4]=ROMS_rx0_rx1_processing(H_simp,double(rmsk_simp),I_simp,double(imsk_simp),strg_file,[0 6]); 

%     if check_ice_mask_consistency(I2,imsk2,['after processing with simplified mask'])==1
%         figure;imagesc(imsk2==1&I2==0);title('subdomain inconsistencies imask==1 & I==0')
%         return
%     end
    if check_all(H,I,lmask,rmask,imask,'3b) after rx0/1 processing')~=0
        return
    end
    % fill processed areas in original fields, implicit desimplify masks
    I2(imsk_simp==1)=I4(imsk_simp==1);
    H2(rmsk_simp==1)=H4(rmsk_simp==1);
    
    if proc_subdomain==1
        H1(crp_ind(1):crp_ind(2),crp_ind(3):crp_ind(4))=H2;
        I1(crp_ind(1):crp_ind(2),crp_ind(3):crp_ind(4))=I2;
        H=H1;
        I=I1;
        if check_all(H,I,lmask,rmask,imask,'3b) after merging processed subdomain with full domain')~=0
            return
        end
        
        disp('== proc stage 3b - finished processing on sub domain')
    else
       H=H2;
       I=I2;
    end
%     
%     % check for NaN
%     if check4NaN(H,I,'3)')
%         return
%     end
%     if check_ice_mask_consistency(I,imask,['after bathy & draft processing '])==1 || ...
%             check_masks(lmask,rmask,imask,0,' After bathy & draft processing ')==1
%         return
%     end
    
    % clean up
    clear H1 I1 H2 I2 H_simp I_simp H4 I4 rmsk imsk
    disp(' ')
   
    % store parameters  -----------------------------------
    H_his(:,:,his_cnt) = H;
    I_his(:,:,his_cnt) = I;
    ID_oce2ice_his(his_cnt) = {ID_oce2ice};
    MSK_his(:,:,1:4,his_cnt) = cat(3,lmask,imask,rmask,omask);
    save(strg_file,'H_his','I_his','MSK_his','ID_oce2ice_his','data_source')
    disp(['Saving masks and fields after I&H processing stage ' num2str(his_cnt)])
elseif proc_stage==4
    disp(['Loading after bathy processing stage.'])
    load(strg_file)
    H = H_his(:,:,his_cnt);
    I = I_his(:,:,his_cnt);
    ID_oce2ice = C2M(ID_oce2ice_his(his_cnt));
    lmask = MSK_his(:,:,1,his_cnt);
    imask = MSK_his(:,:,2,his_cnt);
    rmask = MSK_his(:,:,3,his_cnt);
    omask = MSK_his(:,:,4,his_cnt);
end

%% 3b) simplify masks
his_cnt = his_cnt+1;
if proc_stage<5
    disp(' ')
    disp('== proc stage 3b [nothing] ================================================')
    % check for NaN
    if check4NaN(H,I,'3b)')
        return
    end
    
    % this removes any grounded ice sheets
%     WC=H+I;
%     %WC(rmask==0)=NaN;
%     rmask(WC<Wc_min)=0;
%     imask(WC<Wc_min)=0;
%     H(WC<Wc_min)=Wc_min;
%     I(WC<Wc_min)=0;
%     
%     lmask(rmask==0)=1; 
    
    ROMS_rx0_rx1_processing(H,double(rmask),I,double(imask),strg_file,[0]); 
    % store parameters after rx1 ------------------------------------------
    H_his(:,:,his_cnt) = H;
    I_his(:,:,his_cnt) = I;
    ID_oce2ice_his(his_cnt) = {ID_oce2ice};
    MSK_his(:,:,1:4,his_cnt) = cat(3,lmask,imask,rmask,omask);
    save(strg_file,'H_his','I_his','MSK_his','ID_oce2ice_his','data_source')
    disp(['Saving masks and fields after simplify mask and wc check stage ' num2str(his_cnt)])
elseif proc_stage==5
    disp(['Loading after simplify masks.'])
    load(strg_file)
    H = H_his(:,:,his_cnt);
    I = I_his(:,:,his_cnt);
    ID_oce2ice = C2M(ID_oce2ice_his(his_cnt));
    lmask = MSK_his(:,:,1,his_cnt);
    imask = MSK_his(:,:,2,his_cnt);
    rmask = MSK_his(:,:,3,his_cnt);
    omask = MSK_his(:,:,4,his_cnt);
end

%% 3d rx0 shapiro filter both bathymetry
his_cnt = his_cnt+1;
if proc_stage<6
    disp(' ')
    disp('== proc stage 3d [nothing] ================================================')
    %% check mask consistency
    if check_masks(lmask,rmask,imask)==1
        return
    else
        % update ocean mask
        omask = (lmask==0&imask==0);
        if mask_control_plots==1
             plot_masks(lmask,rmask,imask,('3d) [does nothting]'));           
        end
    end

    % check for NaN
    if check4NaN(H,I,'3d)')
        return
    end
    
    % store parameters after rx0 ------------------------------------------
    H_his(:,:,his_cnt) = H;
    I_his(:,:,his_cnt) = I;
    ID_oce2ice_his(his_cnt) = {ID_oce2ice};
    MSK_his(:,:,1:4,his_cnt) = cat(3,lmask,imask,rmask,omask);
    save(strg_file,'H_his','I_his','MSK_his','ID_oce2ice_his','data_source')
    disp(['Saving masks and fields after [nothing] ' num2str(his_cnt)])
elseif proc_stage==6
    disp(['Loading after smooth_rx0 stage.'])
    load(strg_file)
    H = H_his(:,:,his_cnt);
    I = I_his(:,:,his_cnt);
    ID_oce2ice = C2M(ID_oce2ice_his(his_cnt));
    lmask = MSK_his(:,:,1,his_cnt);
    imask = MSK_his(:,:,2,his_cnt);
    rmask = MSK_his(:,:,3,his_cnt);
    omask = MSK_his(:,:,4,his_cnt);
end


%% 3e) deal with Wc in open ocean and under ice shelves
his_cnt = his_cnt+1;
if proc_stage<7
    disp(' ')
    disp('== proc stage 3e ================================================')
    
    % check for NaN
    if check4NaN(H,I,'3e)')
        return
    end
 
    % 6) check for new narrows
    disp(['== proc stage 3e) - check again for narrows and lakes on rmask'])
   
    [ lmask,rmask,imask,omask,ID_oce2ice ] = remove_rmask_narrows( lmask,rmask,imask,omask );
    
    %% 1e) make I and H consistent in areas of grounded ice sheet 
    %% (disabled for now, instead H is set to min depth and I to zero)
    grnd_sheet_map = imask==1&rmask==0&lmask==0;
    % I(grnd_sheet_map) =  -H(grnd_sheet_map);
    I(grnd_sheet_map) = 0;
    H(grnd_sheet_map) = 50;
    
    ROMS_rx0_rx1_processing(H,double(rmask),I,double(imask),strg_file,[0]); 
    
    % store parameters after Wc correction ------------------------------------
    H_his(:,:,his_cnt) = H;
    I_his(:,:,his_cnt) = I;
    ID_oce2ice_his(his_cnt) = {ID_oce2ice};
    MSK_his(:,:,1:4,his_cnt) = cat(3,lmask,imask,rmask,omask);
    save(strg_file,'H_his','I_his','MSK_his','ID_oce2ice_his','data_source')
    disp(['Saving masks and fields after water column correction ' num2str(his_cnt)])
elseif proc_stage==7
    disp(['Loading after water column correction stage.'])
    load(strg_file)
    H = H_his(:,:,his_cnt);
    I = I_his(:,:,his_cnt);
    ID_oce2ice = C2M(ID_oce2ice_his(his_cnt));
    lmask = MSK_his(:,:,1,his_cnt);
    imask = MSK_his(:,:,2,his_cnt);
    rmask = MSK_his(:,:,3,his_cnt);
    omask = MSK_his(:,:,4,his_cnt);
end


%% 3f) deal with individual problematic regions
his_cnt = his_cnt+1;
if proc_stage<8 
    disp(' ')
    disp('== proc stage 3f ================================================')
    
    % check for NaN
    if check4NaN(H,I,'3f)')
        return
    end
    % remove faulty spots
  
    % check for NaN
    if check4NaN(H,I,'3f')
        return
    end  
    
    % store parameters after special treatment ----------------------------
    H_his(:,:,his_cnt) = H;
    I_his(:,:,his_cnt) = I;
    ID_oce2ice_his(his_cnt) = {ID_oce2ice};
    MSK_his(:,:,1:4,his_cnt) = cat(3,lmask,imask,rmask,omask);
    save(strg_file,'H_his','I_his','MSK_his','ID_oce2ice_his','data_source')
    disp(['Saving masks and fields after special treatment ' num2str(his_cnt)])
elseif proc_stage==8
    disp(['Loading after special treatment.'])
    load(strg_file)
    H = H_his(:,:,his_cnt);
    I = I_his(:,:,his_cnt);
    ID_oce2ice = C2M(ID_oce2ice_his(his_cnt));
    lmask = MSK_his(:,:,1,his_cnt);
    imask = MSK_his(:,:,2,his_cnt);
    rmask = MSK_his(:,:,3,his_cnt);
    omask = MSK_his(:,:,4,his_cnt);
end

%% check masks 
% 1:land | 2:open ocean | 4: grounded ice sheet | 6: ice shelf
if check_masks(lmask,rmask,imask)==1
    return
else
    % update ocean mask
    omask = (lmask==0&imask==0);
    if mask_control_plots==1
        plot_masks(lmask,rmask,imask,('4) before sponging and merging'));
    end

end

%% 4) add sponge area or merge with alien bathymetry, test coordinate system, pick bad spots
%% 4a) add sponge area
disp(' ')
disp('== proc stage 4a ================================================')
% check for NaN
if check4NaN(H,I,'4)')
    return
end

[H, tmp_mask, rel_pts]= add_sponge_sj(H, avg_filter, n_sponge , sponge_layer, sponge_dir,3,rmask);
disp(['== proc stage 4a - adding sponge in directions ' num2str(sponge_dir)])
% update land&ocean mask for sponge layer rmask manipulations
lmask(tmp_mask==1&rmask==0)=0;
lmask(tmp_mask==0&rmask==1)=1;
omask = (lmask==0&imask==0);
% update imask for sponge layer rmask manipulation
imask(tmp_mask==0&rmask==1) = 0;
% assign new rmask
rmask=tmp_mask; 

% check masks 
% 1:land | 2:open ocean | 4: grounded ice sheet | 6: ice shelf
if check_masks(lmask,rmask,imask)==1
    return
else
    % update ocean mask
    omask = (lmask==0&imask==0);
    if mask_control_plots==1
        plot_masks(lmask,rmask,imask,'after sponging');
    end
end
clear tmp_mask rel_mask tmp_H

% check for NaN
if check4NaN(H,I,'4a) after sponging')
    return
end

% check rx factors
ROMS_rx0_rx1_processing(H,double(rmask),I,double(imask),strg_file,[0]); 
    

%% 4b) merge with alien bathy & limit depth
disp(' ')
disp('== proc stage 4b ================================================')
% check for NaN
if check4NaN(H,I,'4b) before merging bathy and limiting depth')
    return
end


%mrg_case = 'SOCS_V21';
mrg_sim  = sim_numbr;
[~,mrg_prep] = set_machine_path(mrg_sim,4);
%mrg_file = ['~/08_data/' mrg_sim '/input/' mrg_sim '_grd_' mrg_case '.nc'];
mrg_file = [MRG_PTH mrg_sim '_grd_' mrg_case '.nc'];
mrg_mpfl = [mrg_prep 'pre_data/mapping_' mrg_case '.mat']; % mapping file
disp(['merging with ' mrg_file])

% set up blend mask
load(mrg_mpfl,'RSSM_regions');
blnd_msk = double(RSSM_regions(5).reg_ind)';
tmp = 1-make_index_boundary_V01(blnd_msk,[1 0],0.0,10);
blnd_msk(isfinite(tmp)) = tmp(isfinite(tmp));

% load alien bathy
alH = ncread(mrg_file,'h');
alI = ncread(mrg_file,'zice');
alimask = ncread(mrg_file,'mask_ice');
alrmask = ncread(mrg_file,'mask_rho');

I = I.*blnd_msk + alI.*(1-blnd_msk);
H = H.*blnd_msk + alH.*(1-blnd_msk);

imask(blnd_msk==0) = alimask(blnd_msk==0);
rmask(blnd_msk==0) = alrmask(blnd_msk==0);

% treat masks (since we are not carrying ice sheet at the moment lmask and
% imask need updating once we decide to represent proper on-land configuration) 
%imask = imask.*blnd_msk + aimask.*(1-blnd_msk);
%rmask = rmask.*blnd_msk + armask.*(1-blnd_msk);
%imask = I~=0;

omask = rmask==1&imask==0;
lmask = rmask==0&imask==0;

if check_masks(lmask,rmask,imask)==1
    return
else
    % update ocean mask
    omask = (lmask==0&imask==0);
    if mask_control_plots==1
        plot_masks(lmask,rmask,imask,'after merging');
    end
end
if check4NaN(H,I,'4b) merging to alien bathymetry')
    return
end

% check rx factors
ROMS_rx0_rx1_processing(H,double(rmask),I,double(imask),strg_file,[0]); 


%% check for remaining NaN's
disp(['Removing remaining NaNs in H : ' num2str(sumsum(isnan(H))) ])
H(isnan(H))=Wc_min;

%% picking for bad spots
if strcmpi(pick_spots, 'on')
    shw_bnd = [1 N_lon 1 N_lat];
    disp('plot bathy to pick spots in bathymetry')
    figure('Position',[1 1 600 800]);
    tmp_mask = rmask;
    tmp_mask(I~=0) = 0;
    imagesc(tmp_mask(shw_bnd(1):shw_bnd(2),shw_bnd(3):shw_bnd(4)));
    title('mask maksing out ice shelf')
    caxis([-1 1])
    figure('Position',[1 1 600 800]);
    imagesc(H(shw_bnd(1):shw_bnd(2),shw_bnd(3):shw_bnd(4)));
    title('bathy')
    colorbar
    figure('Position',[1 1 600 800]);
    imagesc(I(shw_bnd(1):shw_bnd(2),shw_bnd(3):shw_bnd(4)));
    title('ice draft')
    colorbar
    figure('Position',[1 1 600 800]);
    imagesc(rmask(shw_bnd(1):shw_bnd(2),shw_bnd(3):shw_bnd(4)));
    title('rmask')
    caxis([-1 1])
    figure('Position',[1 1 600 800]);
    imagesc(imask(shw_bnd(1):shw_bnd(2),shw_bnd(3):shw_bnd(4)));
    title('i mask')
    caxis([-1 1])
    return
end

%% end of bathy manipulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 4b) test plotting sigma coordinates system
disp(' ')
disp('== proc stage 4b ================================================')
if strcmpi(plt_sig_layers, 'on')
    kgrid = 1;  % 0: rho points being calculated
    column=1; % row section
    LON_SRC = ncread(grd_file,'lon_rho')';
    LAT_SRC = ncread(grd_file,'lat_rho')';
    for i=round(N_lat/2):round(N_lat/2)
        
        [dum,sc_r,Cs_r]=scoord_zice(H(:,:), I(:,:), LON_SRC(:,:), LAT_SRC(:,:), ARVD.Vtransform, ARVD.Vstretching, ...
            ARVD.ThetaS, ARVD.ThetaB, ARVD.hc, ARVD.N, kgrid, column, i, 0);
        %Z(:,:,i)=dum';
        Z=dum';
    end
    %Z = flipdim(permute(Z,[2,3,1]),3);
    
    figure
    hold on
    for jj=1:size(Z,1)
        plot(squeeze(Z(jj,:)),'blue');
    end
    plot(squeeze(-H(i,:)),'black');
    hold off
    title('sigma coordinates')
end

%% 4c) calculating remaining parameters
disp(' ')
disp('== proc stage 4c ================================================')
% transpose those fields for processing to [vertical horizontal] dimension
% all resulting coordinate and mask fields need to be transposed back for
% filling in nc file
H = H';
I = I';
rmask = rmask';
imask = imask';
lmask = lmask';
omask = omask';

smask = imask==1&rmask==1; % establish ice shelf mask
% update water column one last time and set to NaN where no ocean
IS = I;
IS(smask==0)=0;
% ROMS requires H to be finite everywhere, set to minimum depth over land
H(rmask==0)=B.Wc_min;
I(rmask==0)=0;

% ROMS requires Wc=I+H > 0. Therefore I sits Wc_min above H.
Wc = H+I;
disp(['Spots shallower than Wc min: ' num2str(sumsum(Wc<Wc_min)) '. '])
%I(Wc<Wc_min&rmask==1)=-(H(Wc<Wc_min&rmask==1)-Wc_min);
%H(Wc<Wc_min&rmask==1)= -I(Wc<Wc_min&rmask==1)+Wc_min;
H(Wc<Wc_min)= -I(Wc<Wc_min)+Wc_min;


Wc(lmask==1)=NaN;
depthmax = max(max(H));
ROMS_rx0_rx1_processing(H,double(rmask),I,double(imask),strg_file,[0 ]);
%% 4d) computing cfl condition
disp(' ')
disp('== proc stage 4d ================================================')
tmp_pm = ncread(grd_file,'pm')';
tmp_pn = ncread(grd_file,'pn')';

CFL_con_x = 1 ./ (tmp_pm(rmask==1) .* sqrt(H(rmask==1).*9.81));
CFL_con_y = 1 ./ (tmp_pn(rmask==1) .* sqrt(H(rmask==1).*9.81));

mask_psi = rmask(2:end,2:end).*rmask(2:end,1:end-1)...
    .*rmask(1:end-1,2:end).*rmask(1:end-1,1:end-1);

mask_u = rmask(:,2:end).*rmask(:,1:end-1);
mask_v = rmask(2:end,:).*rmask(1:end-1,:);

%% 4e) filling data into nc file and .mat file respectively
    

disp(' ')
disp('== proc stage 4e ================================================')
% 2D vars
nc_info = ncinfo(grd_file);
nc_vnme = {nc_info.Variables.Name};

ncwrite(grd_file, 'h',H');

ncwrite(grd_file, 'zice',I')
ncwrite(grd_file, 'mask_rho',double(rmask)');
ncwrite(grd_file, 'mask_psi',mask_psi');
ncwrite(grd_file, 'mask_u',mask_u');
ncwrite(grd_file, 'mask_v',mask_v');

if ~any(contains(nc_vnme,'mask_ice'))
    nccreate(grd_file,'mask_ice','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','double');
    nccreate(grd_file,'mask_oce','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','double');
    nccreate(grd_file,'mask_lnd','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','double');
    nccreate(grd_file,'mask_ish','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','double');
end
ncwrite(grd_file, 'mask_ice',double(imask)');
ncwrite(grd_file, 'mask_oce',double(omask)');
ncwrite(grd_file, 'mask_lnd',double(lmask)');
ncwrite(grd_file, 'mask_ish',double(smask)');
[~,amask] = check_masks(lmask,rmask,imask,1);
if ~any(contains(nc_vnme,'mask_all'))
    nccreate(grd_file,'mask_all','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','single');
    ncwriteatt(grd_file,'mask_all','long_name','1:land|2:open ocean|4:grounded ice sheet|6: ice shelf');
end
ncwrite(grd_file, 'mask_all',double(amask)');

if ~any(contains(nc_vnme,'Wc'))
    nccreate(grd_file, 'Wc','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','double');
    nccreate(grd_file, 'IceShelf','Dimensions',{'xi_rho' 'eta_rho'},'Datatype','double');
end
ncwrite(grd_file,'Wc',Wc');
ncwrite(grd_file,'IceShelf',IS');
if ~any(contains(nc_vnme,'Vstretching'))
    nccreate(grd_file,'Vstretching','Datatype','single');
    nccreate(grd_file,'Vtransform','Datatype','single');
    nccreate(grd_file,'theta_s','Datatype','single');
    nccreate(grd_file,'theta_b','Datatype','single')
    nccreate(grd_file,'N','Datatype','single');
    nccreate(grd_file,'hc','Datatype','single');
end
ncwrite(grd_file,'Vstretching',ARVD.Vstretching);
ncwrite(grd_file,'Vtransform',ARVD.Vtransform);
ncwrite(grd_file,'theta_s',ARVD.ThetaS);
ncwrite(grd_file,'theta_b',ARVD.ThetaB)
ncwrite(grd_file,'N',ARVD.N);
ncwrite(grd_file,'hc',ARVD.hc);

if new_grd_fle==1
    ncwriteatt(grd_file, '/','data source', data_source);
    ncwriteatt(grd_file, '/','R_Haidvogel, R_haney (h,zice)', num2str([rx0 rx1_h rx1_i]));
    ncwriteatt(grd_file, '/','grid file name', [sim_numbr '_grd_' sim_case '.nc']);
end
ncwriteatt(grd_file, '/','cfl criteria [s]', [num2str(round(minmin([CFL_con_y(:) CFL_con_x(:)])))]);



%% establish trg_grd structure

trg_grd = roms_get_grid_zice(grd_file,[ARVD.ThetaS, ARVD.ThetaB,ARVD.hc,ARVD.N,ARVD.Vtransform,ARVD.Vstretching],0,1,0);%, inifile,0,1);
%trg_grd.Xw84_rho = Xw84_TRG';
%trg_grd.Yw84_rho = Yw84_TRG';
%trg_grd.Neta = size(H,1);
%trg_grd.Nxi = size(H,2);
trg_grd.mask_lnd = lmask;
if ~any(contains(nc_vnme,'z_r'))
    nccreate(grd_file, 'z_r','Dimensions',{'xi_rho' 'eta_rho' 's_rho' trg_grd.N},'Datatype','double');
end
ncwrite(grd_file,'z_r',permute(trg_grd.z_r,[3 2 1]))
if ~any(contains(nc_vnme,'z_w'))
    nccreate(grd_file, 'z_w','Dimensions',{'xi_rho' 'eta_rho' 's_w' trg_grd.N+1},'Datatype','double');
end
ncwrite(grd_file,'z_w',permute(trg_grd.z_w,[3 2 1]))

%% reporting some specs
disp('=================================================================')
disp('some grid specs :')
disp(['min max grid size x/y direction ' num2str(minmax(1./tmp_pm)) ' / ' num2str(minmax(1./tmp_pn)) ]);
disp(['min cfl criteria, barotropic time step in x: ', num2str(minmin(CFL_con_x))]);
disp(['min cfl criteria in y: ', num2str(minmin(CFL_con_y))]);


%% re-establish mapping file, saving trg_grd

%save([pre_path 'pre_data/mapping_' sim_case '.mat'],'trg_grd','-v7.3');
save(map1_file,'trg_grd','-v7.3');
save(map2_file,'trg_grd','-v7.3');
disp(['grid stored in: '])
disp([map1_file])
disp([map2_file])
%% integrate region mapping here
make_region_mapping({sim_case, sim_numbr})
disp(['grid and bathymetry stored in: '])
disp(grd_file)
disp(['building took ' num2str(round_dec(toc(timer0),1)) 's'])

%% clean up for standalone app

%poolobj = gcp('nocreate');
%delete(poolobj);
warning('on', 'all')
close all

end
%% ------------------------------------------------------------------------
%  function definitions



function[H,lmask,rmask]=remove_islands(H,lmask,rmask,margin_width)
global M
tmp_msk = lmask;
tmp_msk(margin_width+1:end-margin_width,margin_width+1:end-margin_width)=1;
[small_isles, c] = islands(tmp_msk==1);

%find and mark all islands one pixel and less than isl_sze
nuke_msk = double((small_isles==0&lmask==1));

c(c(:,2)>=M.isl_sze,:)=[];
for i_isl=1:size(c,1)
    nuke_msk(small_isles==c(i_isl,1))=1;
end
H(nuke_msk==1)=NaN;
% update relevant masks
lmask(nuke_msk==1)=0;
rmask(nuke_msk==1)=1;
%omask(nuke_msk==1)=1;
% extrapolate islands from surrounding bathymetry
H = nan_extrap_V02(H,4,'h','no_plot',3);

disp(['removed ' num2str(sumsum(nuke_msk)+size(c,1)) ' islands '])

end




function[lmask,rmask,imask,accum_del_msk]=remove_lmask_narrows(lmask,rmask,imask)
global M
ii=1;
accum_del_msk = zeros(size(lmask));
while ii<=M.max_iterations
    %% lmask: fill narrows
    
    [del_msk, del_Dmsk ] = find_narrows(lmask==0,M.fill_narrows);
    % update del_msk with the diagonal deletes
    del_msk(del_Dmsk==1)=1;
    
    idel_msk = (del_msk==1&imask==1);
    lmask(del_msk==1)=1;
    rmask(del_msk==1)=0;
    imask(idel_msk)=0;
    accum_del_msk(del_msk==1)=1;
    
    disp('---------------------------------------------------------------')
    disp(['Round ' num2str(ii) ' of ' num2str(M.max_iterations) ' iterations, land2land narrows and holes.'])
    
    
    %% lmask: fill holes
    [lakes, b] = islands(lmask==0);
    
    % find biggest area (the simulated ocean) & and delete from index
    % make first sure it is ocean what we delete!
    
    [~, d]=max(b(:,2).*(b(:,3)==1));
    b(d,:) = [];
    
    N_lks = size(b,1);
    
    % fill pixels on lmask
    n_corr = sumsum(lmask(lakes==0)==0);
    % fill pixels on imask if needed
    ni_corr = sumsum(lmask(lakes==0)==0&imask(lakes==0)==1);
    % update masks
    lmask(lakes==0) = 1;
    rmask(lakes==0) = 0;
    imask(lakes==0&imask==1) = 0;
    % fill areas of multiple pixels
    for i=1:N_lks
        if sumsum(lmask(lakes == b(i,1)))==0
            lmask(lakes == b(i,1)) = 1;
            rmask(lakes == b(i,1)) = 0;
            
            n_corr=n_corr+1;
            if sumsum(imask(lakes == b(i,1))==1)>0
                imask(lakes == b(i,1)&imask==1) = 0;
                ni_corr=ni_corr+1;
            end
        end
    end
    disp(['Removed ' num2str(sumsum(del_msk==1)) ' spots of narrows and ' num2str(n_corr) ' holes on lmask.'])
    disp(['Of narrows ' num2str(sumsum(idel_msk==1)) ' were ice; of holes ' num2str(ni_corr) ' were fully or partly ice. Removed on imask.'])
    if sumsum(del_msk==1)==0 && n_corr==0
        ii=M.max_iterations+1;
    else
        ii=ii+1;
    end
    
end
end


function[fh]=plot_masks(lmask,rmask,imask,tit_txt)

fh= figure('Position',[20 20 800 800]);

gap = [.05 .01];
ygap = [.01 .05];
xgap = [.06 .01];
sub_plot = @(m,n,p) subtightplot (m, n, p, gap,ygap,xgap);

sub_plot(2,2,1)
imagesc(lmask);title(['lmask @ ' tit_txt])

sub_plot(2,2,2)
imagesc(rmask);title(['rmask @ ' tit_txt])
set(gca,'yTickLabel',[])

sub_plot(2,2,3)
imagesc(imask);title(['imask @ ' tit_txt])
set(gca,'xTickLabel',[])

[~,cmask] = check_masks(lmask,rmask,imask,1);
sub_plot(2,2,4)
imagesc(cmask);title(['cmask @ ' tit_txt])
set(gca,'yTickLabel',[],'xTickLabel',[])



end
