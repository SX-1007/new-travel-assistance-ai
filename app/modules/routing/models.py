"""Routing and public map-domain models."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from app.shared.types import ensure_finite_number


class RouteData(BaseModel):
    """Routing metadata for one travel profile."""

    model_config = ConfigDict(extra="forbid")

    distance_km: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    duration_mins: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    @field_validator("distance_km", "duration_mins", mode="before")
    @classmethod
    def _metrics_are_finite(
        cls,
        value: Any,
        info: Any,
    ) -> Any:
        if value is None:
            return value

        return ensure_finite_number(
            value,
            field_name=info.field_name,
        )


class RouteProfiles(BaseModel):
    """Closed set of supported routing profiles."""

    model_config = ConfigDict(extra="forbid")

    driving: RouteData | None = None
    walking: RouteData | None = None
    cycling: RouteData | None = None


class RouteInfo(BaseModel):
    """Public per-day route summary without provider-private payloads."""

    model_config = ConfigDict(extra="forbid")

    ordered_stops: list[str] = Field(
        default_factory=list,
    )
    profiles: RouteProfiles = Field(
        default_factory=RouteProfiles,
    )


class GeoJSONPointGeometry(BaseModel):
    """A two-dimensional GeoJSON point."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["Point"]
    coordinates: list[float] = Field(
        min_length=2,
        max_length=2,
    )

    @field_validator("coordinates", mode="before")
    @classmethod
    def _coordinates_are_finite(
        cls,
        value: Any,
    ) -> Any:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
        ):
            raise ValueError(
                "Point coordinates must contain longitude and latitude"
            )

        longitude = ensure_finite_number(
            value[0],
            field_name="coordinates",
        )
        latitude = ensure_finite_number(
            value[1],
            field_name="coordinates",
        )

        if not -180 <= longitude <= 180:
            raise ValueError(
                "Point longitude must be between -180 and 180"
            )

        if not -90 <= latitude <= 90:
            raise ValueError(
                "Point latitude must be between -90 and 90"
            )

        return [
            longitude,
            latitude,
        ]


class GeoJSONLineStringGeometry(BaseModel):
    """A public route line containing only its coordinate path."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["LineString"]
    coordinates: list[list[float]] = Field(
        min_length=2,
    )

    @field_validator("coordinates", mode="before")
    @classmethod
    def _coordinates_are_finite(
        cls,
        value: Any,
    ) -> Any:
        if (
            not isinstance(value, (list, tuple))
            or len(value) < 2
        ):
            raise ValueError(
                "LineString coordinates must contain at least two points"
            )

        normalized: list[list[float]] = []

        for point in value:
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
            ):
                raise ValueError(
                    "Each LineString point must contain two coordinates"
                )

            longitude = ensure_finite_number(
                point[0],
                field_name="coordinates",
            )
            latitude = ensure_finite_number(
                point[1],
                field_name="coordinates",
            )

            if not -180 <= longitude <= 180:
                raise ValueError(
                    "LineString longitude must be between -180 and 180"
                )

            if not -90 <= latitude <= 90:
                raise ValueError(
                    "LineString latitude must be between -90 and 90"
                )

            normalized.append(
                [
                    longitude,
                    latitude,
                ]
            )

        return normalized


GeoJSONGeometry = Annotated[
    GeoJSONPointGeometry | GeoJSONLineStringGeometry,
    Field(discriminator="type"),
]


class GeoJSONProperties(BaseModel):
    """Closed properties emitted for waypoint pins and route lines."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None

    type: Literal[
        "activity",
        "attraction",
        "restaurant",
        "hotel",
        "airport",
        "route",
    ]

    order: int | None = Field(
        default=None,
        ge=0,
        strict=True,
    )

    profile: Literal[
        "driving",
        "walking",
        "cycling",
    ] | None = None

    distance_km: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    duration_mins: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    @field_validator(
        "distance_km",
        "duration_mins",
        mode="before",
    )
    @classmethod
    def _metrics_are_finite(
        cls,
        value: Any,
        info: Any,
    ) -> Any:
        if value is None:
            return value

        return ensure_finite_number(
            value,
            field_name=info.field_name,
        )

    @model_serializer(mode="wrap")
    def _preserve_producer_shape(
        self,
        handler: Any,
    ) -> dict[str, Any]:
        serialized = handler(self)

        return {
            key: value
            for key, value in serialized.items()
            if key in self.model_fields_set
        }


class GeoJSONFeature(BaseModel):
    """Closed GeoJSON feature containing only producer-supported fields."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["Feature"]
    geometry: GeoJSONGeometry
    properties: GeoJSONProperties

    @model_validator(mode="after")
    def _geometry_matches_properties(
        self,
    ) -> "GeoJSONFeature":
        route_geometry = (
            self.geometry.type == "LineString"
        )

        route_properties = (
            self.properties.type == "route"
        )

        if route_geometry != route_properties:
            raise ValueError(
                "route properties require LineString geometry"
            )

        property_fields = (
            self.properties.model_fields_set
        )

        if route_properties:
            if property_fields != {
                "type",
                "profile",
                "distance_km",
                "duration_mins",
            }:
                raise ValueError(
                    "route properties must use the exact public shape"
                )

            if (
                self.properties.profile is None
                or self.properties.distance_km is None
                or self.properties.duration_mins is None
            ):
                raise ValueError(
                    "route properties require profile and metrics"
                )

        else:
            if property_fields != {
                "name",
                "type",
                "order",
            }:
                raise ValueError(
                    "point properties must use the exact public shape"
                )

            if (
                not isinstance(
                    self.properties.name,
                    str,
                )
                or not self.properties.name.strip()
            ):
                raise ValueError(
                    "point properties require a non-empty name"
                )

            if self.properties.order is None:
                raise ValueError(
                    "point properties require an integer order"
                )

        return self


class GeoJSONFeatureCollection(BaseModel):
    """Closed daily map payload safe for public serialization."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["FeatureCollection"]

    features: list[GeoJSONFeature] = Field(
        default_factory=list,
    )


def canonicalize_daily_map_keys(
    value: Any,
) -> dict[int, Any]:
    """Normalize positive canonical day keys and reject ambiguous collisions."""

    if not isinstance(value, Mapping):
        raise ValueError(
            "daily maps must be a mapping"
        )

    canonical: dict[int, Any] = {}

    for raw_key, daily_map in value.items():
        if isinstance(raw_key, bool):
            raise ValueError(
                "daily map keys must be positive integers"
            )

        if isinstance(raw_key, int):
            day = raw_key

        elif (
            isinstance(raw_key, str)
            and raw_key.isascii()
            and raw_key.isdigit()
            and raw_key == str(int(raw_key))
        ):
            day = int(raw_key)

        else:
            raise ValueError(
                "daily map keys must be canonical positive integers"
            )

        if day <= 0:
            raise ValueError(
                "daily map keys must be positive integers"
            )

        if day in canonical:
            raise ValueError(
                "daily map keys collide after normalization"
            )

        canonical[day] = daily_map

    return canonical


def canonicalize_public_daily_maps(
    value: Any,
) -> dict[int, Any]:
    """Canonicalize raw maps while preserving already-projected map objects."""

    if isinstance(value, dict) and all(
        isinstance(day, int)
        and not isinstance(day, bool)
        and day > 0
        and isinstance(
            daily_map,
            GeoJSONFeatureCollection,
        )
        for day, daily_map in value.items()
    ):
        return value

    return canonicalize_daily_map_keys(
        value
    )


__all__ = [
    "RouteData",
    "RouteProfiles",
    "RouteInfo",
    "GeoJSONPointGeometry",
    "GeoJSONLineStringGeometry",
    "GeoJSONGeometry",
    "GeoJSONProperties",
    "GeoJSONFeature",
    "GeoJSONFeatureCollection",
    "canonicalize_daily_map_keys",
    "canonicalize_public_daily_maps",
]