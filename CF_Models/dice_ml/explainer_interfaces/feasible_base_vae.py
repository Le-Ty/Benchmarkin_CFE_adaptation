#General Imports
import numpy as np
import random
import collections
import timeit
import copy

#Dice Imports
from CF_Models.dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from CF_Models.dice_ml import diverse_counterfactuals as exp
from CF_Models.dice_ml.utils.sample_architecture.vae_model import CF_VAE
from CF_Models.dice_ml.utils.helpers import get_base_gen_cf_initialization

#Pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class FeasibleBaseVAE(ExplainerBase):

    def __init__(self, data_interface, model_interface, max_iter=1000, **kwargs):
        """
        :param data_interface: an interface class to data related params
        :param model_interface: an interface class to access trained ML model
        :param max_iter: int > 0; max number of search iterations
        """ 
        
        # initiating data related parameters
        super().__init__(data_interface) 
        
        #Black Box ML Model to be explained
        self.pred_model= model_interface.model        
                
        #Hyperparam 
        self.encoded_size=kwargs['encoded_size']
        self.learning_rate= kwargs['lr']
        self.batch_size= kwargs['batch_size']
        self.validity_reg= kwargs['validity_reg']
        self.margin= kwargs['margin']
        self.epochs= kwargs['epochs']
        self.wm1= kwargs['wm1']
        self.wm2= kwargs['wm2']
        self.wm3= kwargs['wm3']
        self.max_iter = max_iter
        
        #Initializing parameters for the DiceBaseGenCF  
        self.vae_train_dataset, self.vae_val_dataset, self.vae_test_dataset, self.normalise_weights, self.cf_vae, self.cf_vae_optimizer= get_base_gen_cf_initialization( self.data_interface, self.encoded_size, self.cont_minx, self.cont_maxx, self.margin, self.validity_reg, self.epochs, self.wm1, self.wm2, self.wm3, self.learning_rate ) 
        
        #Data paths
        self.base_model_dir= 'CF_Models/dice_ml/utils/sample_trained_models/'
        # self.base_model_dir= '../../dice_ml/utils/sample_trained_models/'
        self.save_path=self.base_model_dir+ self.data_interface.data_name +'-margin-' + str(self.margin) + '-validity_reg-'+ str(self.validity_reg) + '-epoch-' + str(self.epochs) + '-' + 'base-gen' + '.pth'

    
    def compute_loss( self, model_out, x, target_label ): 

        em = model_out['em']
        ev = model_out['ev']
        z  = model_out['z']
        dm = model_out['x_pred']
        mc_samples = model_out['mc_samples']
        #KL Divergence
        kl_divergence = 0.5*torch.mean( em**2 +ev - torch.log(ev) - 1, axis=1 ) 

        #Reconstruction Term
        #Proximity: L1 Loss
        x_pred = dm[0]       
        s= self.cf_vae.encoded_start_cat
        recon_err = -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
        for key in self.normalise_weights.keys():
            recon_err+= -(self.normalise_weights[key][1] - self.normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

        # Sum to 1 over the categorical indexes of a feature
        for v in self.cf_vae.encoded_categorical_feature_indexes:
            temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
            recon_err += temp

        #Validity
        temp_logits = self.tensor_to_logit(self.pred_model(x_pred).unsqueeze(1))
        # temp_logits = self.pred_model(x_pred).unsqueeze(1)
        validity_loss= torch.zeros(1)                                       
        temp_1= temp_logits[target_label==1,:]
        temp_0= temp_logits[target_label==0,:]
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]) - F.sigmoid(temp_1[:,0]), torch.tensor(-1), self.margin, reduction='mean')
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]) - F.sigmoid(temp_0[:,1]), torch.tensor(-1), self.margin, reduction='mean')

        for i in range(1,mc_samples):
            x_pred = dm[i]       

            recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
            for key in self.normalise_weights.keys():
                recon_err+= -(self.normalise_weights[key][1] - self.normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

            # Sum to 1 over the categorical indexes of a feature
            for v in self.cf_vae.encoded_categorical_feature_indexes:
                temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
                recon_err += temp

            #Validity
            temp_logits = self.tensor_to_logit(self.pred_model(x_pred).unsqueeze(1))
            # temp_logits = self.pred_model(x_pred)
            temp_1= temp_logits[target_label==1,:]
            temp_0= temp_logits[target_label==0,:]
            validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]) - F.sigmoid(temp_1[:,0]), torch.tensor(-1), self.margin, reduction='mean')
            validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]) - F.sigmoid(temp_0[:,1]), torch.tensor(-1), self.margin, reduction='mean')

        recon_err = recon_err / mc_samples
        validity_loss = -1*self.validity_reg*validity_loss/mc_samples

        print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence), ' Validity: ', -validity_loss)
        return -torch.mean(recon_err - kl_divergence) - validity_loss

    
    def train(self, pre_trained=False):
        '''        
        pre_trained: Bool Variable to check whether pre trained model exists to avoid training again        
        '''
        
        if pre_trained:
            self.cf_vae.load_state_dict(torch.load(self.save_path))
            self.cf_vae.eval()
            return 
        
        ##TODO: Handling such dataset specific constraints in a more general way
        # CF Generation for only low to high income data points
        self.vae_train_dataset= self.vae_train_dataset[self.vae_train_dataset[:,-1]==0,:]
        self.vae_val_dataset= self.vae_val_dataset[self.vae_val_dataset[:,-1]==0,:]

        #Removing the outcome variable from the datasets
        self.vae_train_feat= self.vae_train_dataset[:,:-1]
        self.vae_val_feat= self.vae_val_dataset[:,:-1]
        
        for epoch in range(self.epochs):
            batch_num=0
            train_loss= 0.0
            train_size=0
            
            train_dataset= torch.tensor(self.vae_train_feat).float()
            train_dataset= torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for train_x in enumerate(train_dataset):
                self.cf_vae_optimizer.zero_grad()
                
                train_x= train_x[1]
                # Warum hier 1-argmax? Prediction ist immer Spaltenvektor, damit wäre argmax immer 0!
                temp_logits = self.tensor_to_logit(self.pred_model(train_x).unsqueeze(1))
                train_y= 1.0-torch.argmax( temp_logits, dim=1 )
                # train_y= 1.0-torch.argmax( self.pred_model(train_x).unsqueeze(1), dim=1 )
                train_size+= train_x.shape[0]
                
                out= self.cf_vae(train_x, train_y)
                loss= self.compute_loss( out, train_x, train_y )
                loss.backward()
                train_loss += loss.item()
                self.cf_vae_optimizer.step()
                
                batch_num+=1
                
            ret= loss/batch_num
            print('Train Avg Loss: ', ret, train_size )
            
            #Save the model after training every 10 epochs and at the last epoch
            if (epoch!=0 and epoch%10==0) or epoch==self.epochs-1:
                torch.save(self.cf_vae.state_dict(), self.save_path)                   
            
    #The input arguments for this function same as the one defined for Diverse CF    
    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite"  ):

        ## Loading the latest trained CFVAE model
        self.cf_vae.load_state_dict(torch.load(self.save_path))
        self.cf_vae.eval()
        
        # Converting query_instance into numpy array
        query_instance_org= query_instance
        
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance = np.array([query_instance.iloc[0].values])
        
        print(query_instance.shape[0])
        if  query_instance.shape[0] > self.batch_size:
            test_dataset= np.array_split( query_instance, query_instance.shape[0]//self.batch_size ,axis=0 )
        else:
            test_dataset= [ query_instance ] 
        final_gen_cf=[]
        final_cf_pred=[]
        final_test_pred=[]
        for i in range(len(query_instance)):
            train_x = test_dataset[i]
            train_x= torch.tensor( train_x ).float()
            temp_logits = self.tensor_to_logit(self.pred_model(train_x).unsqueeze(0))
            train_y = torch.argmax( temp_logits, dim=1 )
            # train_y = torch.argmax( self.pred_model(train_x), dim=1 )

            curr_gen_cf=[]
            curr_cf_pred=[]            
            curr_test_pred= train_y.numpy()
            
            for cf_count in range(total_CFs):                                
                recon_err, kl_err, x_true, x_pred, cf_label = self.cf_vae.compute_elbo( train_x, 1.0-train_y, self.pred_model )
                
                count = 0
                step = 1
                max_iter_reached = False
                
                while(cf_label==train_y):
                    #print(cf_label,train_y)
                    # We added max_iter condition: if run out of max_iter, return >>nan<<
                    if count > self.max_iter:
                        max_iter_reached = True
                        break
                    else:
                        count = count + step
                        recon_err, kl_err, x_true, x_pred, cf_label = self.cf_vae.compute_elbo(train_x, 1.0-train_y, self.pred_model)
                    
                if not max_iter_reached:
                    x_pred= x_pred.detach().numpy()
                    #Converting mixed scores into one hot feature representations
                    for v in self.cf_vae.encoded_categorical_feature_indexes:
                        curr_max= x_pred[:, v[0]]
                        curr_max_idx= v[0]
                        for idx in v:
                            if curr_max < x_pred[:, idx]:
                                curr_max= x_pred[:, idx]
                                curr_max_idx= idx
                        for idx in v:
                            if idx==curr_max_idx:
                                x_pred[:, idx]=1
                            else:
                                x_pred[:, idx]=0
                else:
                    x_pred = x_pred.detach().numpy()
                    x_pred[:] = np.nan  # assign np.nans
                    x_pred = x_pred.reshape(1,-1)
                    
                cf_label= cf_label.detach().numpy()
                cf_label= np.reshape(cf_label, (cf_label.shape[0],1))
                
                curr_gen_cf.append( x_pred )
                curr_cf_pred.append( cf_label )
                
            final_gen_cf.append(curr_gen_cf)
            final_cf_pred.append(curr_cf_pred)
            final_test_pred.append(curr_test_pred)
        
        #CF Gen out
        result={}
        result['query-instance']= query_instance[0]
        result['test-pred']= final_test_pred[0][0]
        result['CF']= final_gen_cf[0]
        result['CF-Pred']= final_cf_pred[0]
        
        # Adding empty list for sparse cf gen and pred; adding 'NA' for the posthoc sparsity cofficient
        return exp.CounterfactualExamples(self.data_interface, result['query-instance'], result['test-pred'], result['CF'], result['CF-Pred'],  posthoc_sparsity_param=None)


    def tensor_to_logit(self, prediction):
        temp_model_out = prediction.detach().numpy()
        temp_mod_idx = np.round(temp_model_out)
        temp_model_logits = np.zeros((prediction.shape[0], 2))
        for i in range(temp_mod_idx.shape[0]):
            temp_model_logits[i][int(temp_mod_idx[i])] = temp_model_out[i]

        return torch.from_numpy(temp_model_logits)
