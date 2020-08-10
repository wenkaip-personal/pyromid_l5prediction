from pytorch_lightning import Callback

class VizReconstructions(Callback):

    def __init__(self, serialize_input: bool=True):
        """
        This callback visualizes the reconstruction :math:`\hat{x}` of the input
        data :math:`x`. 
        
        .. note:: It is meant to be used with an autoencoder that has the
                 following modules:
                    `model.encoder(x) -> z`

                    `model.decoder(z) -> x_hat`
                    
                 The model must also have a attributes:
                    `self.last_batch` 

                    `self.img_shape` (for proper image sizes)

                 `batch` must be of the format `(x,y)` where `x` is the data to reconstruct. 

        Example::

            from src.callbacks.viz import VizReconstructions
            Trainer(args, callbacks=[VizReconstructions()])

        .. note:: This callback only supports tensorboard at the moment.

        Args:
            serialize_input: Flag to indicate serialization of x when passed to model.

        Authored by:

            - Rosario Scalise

        """
        super().__init__()
        self.serialize_input = serialize_input

    def on_epoch_end(self, trainer, model):
        import torchvision

        x, _ = model.last_batch 

        # get image shape from batch if model doesn't specify self.img_shape
        img_shape = model.img_shape if hasattr(model, 'img_shape') else x.shape[1:]
        cur_batch_size = x.shape[0]

        # forward pass logic 
        x = x.view(x.shape[0],-1) if self.serialize_input else x # preshape x 
        z = model(x) # encode
        x_hat = model.decoder(z) # decode

        # logging
        x_hat = x_hat.view(cur_batch_size, *img_shape)
        grid_x_hat = torchvision.utils.make_grid(x_hat)
        model.logger.experiment.add_image('x_hat', grid_x_hat, model.current_epoch)

class VizStepDict(Callback):

    def __init__(self):
        """
        This callback visualizes all entries in the step_dict['viz'] dictionary.
        This is most useful in a multi-task setting where the user would like to
        visualize multiple groundtruth images and corresponding reconstructions.

        .. note:: It is meant to be used with an autoencoder that has the
                 following method:
                    `_step()` which returns a `step_dict` containing a `'viz'` key

                 The model must also have a attributes:
                    `self.last_batch` 

                    `self.img_shape` (for proper image sizes)

                 `batch` must be of the format `(x,y)` where `x` is the data to reconstruct. 

        Example::
            from src.callbacks.viz import VizStepDict
            Trainer(args, callbacks=[VizStepDict()])

        .. note:: This callback only supports tensorboard at the moment.

        Authored by:

            - Rosario Scalise

        """
        super().__init__()

    def on_epoch_end(self, trainer, model):
        import torchvision

        batch = model.last_batch 
        step_dict = model._step(batch) 

        # Check to see if the step_dict has anything to visualize in it.
        try:
            first_item = next(iter(step_dict['viz'].values())) 
        except Exception as exception:
            print("\n Viz Callback Exception: step_dict['viz'] is empty. Nothing to visualize; exiting callback.")
            return

        # get image shape from batch if model doesn't specify self.img_shape
        img_shape = model.img_shape if hasattr(model, 'img_shape') else x.shape[1:]
        cur_batch_size = first_item.shape[0]

        # logging
        # Iterates and plots all items in step_dict's 'viz' key 
        for k,v in step_dict['viz'].items():
            img_batch = v.view(cur_batch_size, *img_shape)
            grid = torchvision.utils.make_grid(img_batch)
            model.logger.experiment.add_image(str(k), grid, model.current_epoch)
