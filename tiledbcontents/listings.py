"""Functions that list contents on the remote server."""

from typing import Any, List

import tornado.web
from tiledb import cloud

from . import caching
from . import models
from . import paths


def namespace(category: str, namespace: str, *, content: bool = False) -> models.Model:
    """Lists all notebook arrays in a namespace, like an `ls`.

    :param category: The category to list; one of "shared", "owned", or "public"
    :param namespace: The namespace to list
    :param content: True to include contents, False to not.
    :return: A model of the namespace.
    """
    arrays = []
    try:
        listing = caching.ArrayListing.from_cache(category, namespace)
        arrays = listing.arrays()
    except cloud.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            500, "Error listing notebooks in {}: {}".format(namespace, str(e))
        )
    except Exception as e:
        raise tornado.web.HTTPError(
            400,
            "Error listing notebooks in  {}: {}".format(namespace, str(e)),
        )

    model = models.create(
        path=paths.join("cloud", category, namespace),
        type="directory",
    )
    if content:
        # Build model content if asked for
        model["format"] = "json"
        model["content"] = []
        if arrays is not None:
            for notebook in arrays:
                nbmodel = models.create(path=notebook.name)

                # Add notebook extension to name, so jupyterlab will open this
                # as a notebook. It seems to check the extension even though
                # we set the "type" parameter.
                nbmodel["path"] = "cloud/{}/{}/{}{}".format(
                    category, namespace, nbmodel["path"], paths.NOTEBOOK_EXT
                )

                nbmodel["last_modified"] = models.to_utc(notebook.last_accessed)
                # Update namespace directory based on last access notebook
                _maybe_update_last_modified(model, notebook)

                nbmodel["type"] = "notebook"

                if (
                    notebook.allowed_actions is None
                    or "write" not in notebook.allowed_actions
                ):
                    model["writable"] = False
                model["content"].append(nbmodel)

    return model


def category(category: str, *, content: bool = True) -> models.Model:
    """Lists the directories within a category.

    :param category: The category to list; one of "shared", "owned", or "public"
    :param content: True to include contents, False to not.
    :return: A directory model for the category.
    """
    # TODO: This function should be switched to use sidebar data.
    arrays = []
    try:
        arrays = caching.ArrayListing.from_cache(category).arrays()
    except cloud.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            500, "Error listing notebooks in {}: {}".format(category, str(e))
        )
    except Exception as e:
        raise tornado.web.HTTPError(
            400,
            "Error listing notebooks in  {}: {}".format(category, str(e)),
        )

    model = models.create(path=paths.join("cloud", category), type="directory")
    if content:
        model["format"] = "json"
        model["content"] = []
        namespaces = {}
        if category == "owned":
            # For owned, we should always list the user and their
            # organizations. If there is actual notebooks they will be
            # listed below. This base listing is so users can create new
            # notebooks in any of the namespaces they are part of.
            try:
                profile = cloud.client.user_profile()
                namespace_model = models.create(
                    path=paths.join("cloud", category, profile.username),
                    type="directory",
                    format="json",
                )
                namespaces[profile.username] = namespace_model

                for org in profile.organizations:
                    # Don't list public for owned
                    if org.organization_name == "public":
                        continue

                    namespace_model = models.create(
                        path=paths.join("cloud", category, org.organization_name),
                        type="directory",
                        format="json",
                    )

                    namespaces[org.organization_name] = namespace_model

            except cloud.TileDBCloudError as e:
                raise tornado.web.HTTPError(
                    500,
                    "Error listing notebooks in {}: {}".format(category, str(e)),
                )
            except Exception as e:
                raise tornado.web.HTTPError(
                    400,
                    "Error listing notebooks in  {}: {}".format(category, str(e)),
                )

        # If the arrays are non-empty list them out
        if arrays is not None:
            for notebook in arrays:
                if notebook.namespace not in namespaces:
                    namespace_model = models.create(
                        path=paths.join("cloud", category, notebook.namespace),
                        type="directory",
                        format="json",
                        writable=False,
                    )
                    namespaces[notebook.namespace] = namespace_model

                # Update directory based on last access notebook
                _maybe_update_last_modified(model, notebook)

                # Update namespace directory based on last access notebook
                _maybe_update_last_modified(namespaces[notebook.namespace], notebook)

        model["content"] = list(namespaces.values())

    return model


def all_notebooks() -> List[models.Model]:
    """List all notebooks, across all categories."""
    try:
        return [_all_notebooks_in(cat) for cat in caching.CATEGORIES]
    except cloud.TileDBCloudError as e:
        raise tornado.web.HTTPError(
            500, f"Error building cloud notebook info: {e}"
        ) from e
    except Exception as e:
        raise tornado.web.HTTPError(
            500, f"Error building cloud notebook info: {e}"
        ) from e


def _all_notebooks_in(category: str) -> models.Model:
    model = models.create(path=paths.join("cloud", category), type="directory")
    listing = caching.ArrayListing.from_cache(category)
    notebooks = listing.arrays()
    if notebooks:
        model.update(
            format="json",
            content=[],
        )
        for notebook in notebooks:
            nb_model = models.create(
                path=notebook.name,
                type="notebook",
                format="json",
                last_modified=models.to_utc(notebook.last_accessed),
            )
            nb_model["path"] = paths.join(
                category, nb_model["path"] + paths.NOTEBOOK_EXT
            )
            model["content"].append(nb_model)
            _maybe_update_last_modified(model, notebook)
    return model


def _maybe_update_last_modified(model: models.Model, notebook: Any) -> None:
    model["last_modified"] = models.max_present(
        (
            models.to_utc(notebook.last_accessed),
            models.to_utc(model.get("last_modified")),
        )
    )
