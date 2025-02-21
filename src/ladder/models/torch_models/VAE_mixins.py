"""The VAE variants module houses base model definitions for VAE style models."""

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine


class _ModelMixin:
    """Includes shared capabilities for models.

    Methods
    -------
    save(path)
        Saves model parameters to disk.

    load(path, map_location=None)
        Loads model parameters from disk.

    """

    def save(self, path):
        """Saves model parameters to disk.

        Parameters
        ----------
        path : `str`
            Path to save model parameters.
        """
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")

    def load(self, path, map_location=None):
        """Loads model parameters from disk.

        Parameters
        ----------
        path : `str`
            Path to find model parameters. Should not include the extensions `_torch.pth` or `_pyro.pth`.

        map_location : `str`, default: None
            Specifies where the model should be loaded. See :class:`~torch.device` for details.
        """
        pyro.clear_param_store()

        if map_location is None:
            self.load_state_dict(torch.load(path + "_torch.pth"))
            pyro.get_param_store().load(path + "_pyro.pth")

        else:
            self.load_state_dict(
                torch.load(path + "_torch.pth", map_location=map_location)
            )
            pyro.get_param_store().load(path + "_pyro.pth", map_location=map_location)


class _VAEMixin(_ModelMixin):
    """Includes shared capabilities for VAE based models.

    Methods
    -------
    model(*args)
        Generative model for the VAE.

    guide(*args)
        Approximate variational posterior for the VAE.

    """

    @staticmethod
    def _get_input_args(*args):
        return args

    @staticmethod
    def _get_latent_args(*args):
        return args

    @staticmethod
    def _get_output_args(*args):
        return args

    def model(self, *args):
        """Generative model for the base VAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):

            z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
                x.device
            ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

            with poutine.scale(None, self.kl_weight):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            with poutine.scale(None, self.recon_weight):
                pyro.sample(
                    "obs",
                    self._reconstruct(z, *args).to_event(1),
                    obs=self._get_output_args(*args),
                )

    def guide(self, *args):
        """Approximate variational posterior for the base VAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            z_loc, z_scale = self.encoder(x)

            with poutine.scale(None, self.kl_weight):
                pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


class _CCVAEMixin(_VAEMixin):
    """Includes shared capabilities for CCVAE (Continuous CVAE) based models.

    Methods
    -------
    model(*args)
        Generative model for the CCVAE.

    guide(*args)
        Approximate variational posterior for the CCVAE.

    """

    @staticmethod
    def _get_input_args(*args):
        return torch.concatenate(args[:2], dim=-1)

    @staticmethod
    def _get_output_args(*args):
        return args[0]

    @staticmethod
    def _get_label_args(*args):
        return args[1]

    @staticmethod
    def _concat_lat_dims(labels, ref_list, dim):
        idxs = labels.int()
        return (
            torch.tensor(
                np.array(
                    [
                        np.concatenate([[ref_list[num]] * dim for num in elem])
                        for elem in idxs
                    ]
                )
            )
            .type_as(labels)
            .to(labels.device)
        )

    def model(self, *args):
        """Generative model for the CCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        pyro.module(self.__class__.__name__, self)

        z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
            x.device
        ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

        w_loc, w_scale = self._concat_lat_dims(
            y, self.w_locs, self.w_dim
        ), self._concat_lat_dims(y, self.w_scales, self.w_dim)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        with poutine.scale(None, self.recon_weight):
            pyro.sample(
                "obs",
                self._reconstruct(torch.concatenate((z, w), dim=-1)).to_event(1),
                obs=self._get_output_args(*args),
            )

    def guide(self, *args):
        """Approximate variational posterior for the CCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        zw_loc, zw_scale = self.encoder(x)

        z_loc, z_scale, w_loc, w_scale = (
            zw_loc[..., : self.latent_dim],
            zw_scale[..., : self.latent_dim],
            zw_loc[..., self.latent_dim :],
            zw_scale[..., self.latent_dim :],
        )

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        with poutine.scale(None, self.w_kl_weight):
            pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        return z


class _HCCVAEMixin(_CCVAEMixin):
    """Includes shared capabilities for HCCVAE (Hierarchical Continuous CVAE) based models.

    Methods
    -------
    model(*args)
        Generative model for the HCCVAE.

    guide(*args)
        Approximate variational posterior for the HCCVAE.

    """

    @staticmethod
    def _get_input_args(*args):
        return args[0]

    def model(self, *args):
        """Generative model for the HCCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        pyro.module(self.__class__.__name__, self)

        z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
            x.device
        ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        w_loc, w_scale = self._concat_lat_dims(
            y, self.w_locs, self.w_dim
        ), self._concat_lat_dims(y, self.w_scales, self.w_dim)

        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        rho_loc, rho_scale = self.decoder_rho(torch.concatenate((z, w), dim=-1))
        rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

        with poutine.scale(None, self.recon_weight):
            pyro.sample(
                "obs",
                self._reconstruct(rho).to_event(1),
                obs=self._get_output_args(*args),
            )

        return z

    def guide(self, *args):
        """Approximate variational posterior for the HCCVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        rho_loc, rho_scale = self.encoder_rho(x)
        rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

        z_loc, z_scale = self.encoder_z(rho)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        w_loc, w_scale = self.encoder_w(
            torch.concatenate((rho, self._get_label_args(*args)), dim=-1)
        )

        with poutine.scale(None, self.w_kl_weight):
            pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        return z


class _MRVAEMixin(_CCVAEMixin):
    """Includes shared capabilities for MRVAE (Multi Resolution VAE) based models.

    Motivation: https://www.biorxiv.org/content/10.1101/2022.10.04.510898v2

    Methods
    -------
    model(*args)
        Generative model for the MRVAE.

    guide(*args)
        Approximate variational posterior for the MRVAE.

    """

    @staticmethod
    def _get_input_args(*args):
        return args[0]

    def model(self, *args):
        """Generative model for the MRVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
            x.device
        ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        w_loc, w_scale = z, torch.ones((x.shape[0], self.latent_dim)).to(x.device)

        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        with poutine.scale(None, self.recon_weight):
            pyro.sample(
                "obs",
                self._reconstruct(w).to_event(1),
                obs=self._get_output_args(*args),
            )

        return z

    def guide(self, *args):
        """Approximate variational posterior for the MRVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        z_loc, z_scale = self.encoder(x)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        w_loc, w_scale = self.encoder_w(
            torch.concatenate((z, self._get_label_args(*args)), dim=-1)
        )

        with poutine.scale(None, self.w_kl_weight):
            pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        return z


class _AdversarialMixin:
    """Includes shared capabilities for models making use of adversarial information loss.

    Methods
    -------
    adversarial(*args)
        Calculates classifier / adversarial loss from inputs.

    """

    def _adversarial_from_encodings(self, z, y):
        classification_loss_z, attr_track = 0, 0

        for i in pyro.plate("label_dims", len(self.label_dims)):
            classification_loss_z += dist.OneHotCategorical(
                logits=getattr(self, f"classifiers_{i}")(z)
            ).log_prob(y[..., attr_track : attr_track + self.label_dims[i]])

            attr_track = attr_track + self.label_dims[i]

        return classification_loss_z

    def classification(self, *args):
        """Calculates classifier / adversarial loss from inputs."""
        x = self._get_input_args(*args)

        zw_loc, zw_scale = self.encoder(x)
        z_loc, z_scale = (
            zw_loc[..., : self.latent_dim],
            zw_scale[..., : self.latent_dim],
        )

        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        return -1 * self._adversarial_from_encodings(z, self._get_label_args(*args))
