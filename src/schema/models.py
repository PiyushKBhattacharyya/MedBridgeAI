from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# From facility_and_ngo_fields.py
# -------------------------------------------------------------------
class BaseOrganization(BaseModel):
    """Base model containing shared fields between Facility and NGO."""
    name: str = Field(..., description="Official name of the organization")
    phone_numbers: Optional[List[str]] = Field(
        None,
        description="The organization's phone numbers in E164 format (e.g. '+233392022664')",
    )
    email: Optional[str] = Field(None, description="The organization's primary email address")
    websites: Optional[List[str]] = Field(
        None, description="Websites associated with the organization"
    )
    yearEstablished: Optional[int] = Field(
        None, description="The year in which the organization was established"
    )
    acceptsVolunteers: Optional[bool] = Field(
        None, description="Indicates whether the organization accepts clinical volunteers"
    )
    facebookLink: Optional[str] = Field(None, description="URL to the organization's Facebook page")
    twitterLink: Optional[str] = Field(
        None, description="URL to the organization's Twitter profile"
    )
    linkedinLink: Optional[str] = Field(None, description="URL to the organization's LinkedIn page")
    instagramLink: Optional[str] = Field(
        None, description="URL to the organization's Instagram account"
    )
    logo: Optional[str] = Field(None, description="URL linking to the organization's logo image")
    source_doc: Optional[str] = Field(None, description="Source document filename for data lineage")

    # Flattened address fields
    address_line1: Optional[str] = Field(
        None,
        description="Street address only (building number, street name). Do NOT include city, state, or country here.",
    )
    address_line2: Optional[str] = Field(
        None, description="Additional street address information (apartment, suite, building name)"
    )
    address_line3: Optional[str] = Field(None, description="Third line of street address if needed")
    address_city: Optional[str] = Field(
        None,
        description="City or town name of the organization. Parse from comma-separated location strings if needed.",
    )
    address_stateOrRegion: Optional[str] = Field(
        None,
        description="State, region, or province of the organization. Parse from comma-separated location strings if needed.",
    )
    address_zipOrPostcode: Optional[str] = Field(
        None, description="ZIP or postal code of the organization"
    )
    address_country: Optional[str] = Field(
        None,
        description="Full country name of the organization. Always extract if country or country code information is present.",
    )
    address_countryCode: Optional[str] = Field(
        None,
        description="ISO alpha-2 country code of the organization. Derive from country name if needed - this field is REQUIRED when country is known.",
    )
    latitude: Optional[float] = Field(None, description="The latitude coordinate for the organization's location")
    longitude: Optional[float] = Field(None, description="The longitude coordinate for the organization's location")


class Facility(BaseOrganization):
    """Pydantic model for facility structured output extraction."""
    facilityTypeId: Optional[Literal["hospital", "pharmacy", "doctor", "clinic", "dentist"]] = Field(
        None, description="type of facility (only one of these values)"
    )
    operatorTypeId: Optional[Literal["public", "private"]] = Field(
        None, description="Indicates if the facility is privately or publicly operated"
    )
    affiliationTypeIds: Optional[
        List[
            Literal["faith-tradition", "philanthropy-legacy", "community", "academic", "government"]
        ]
    ] = Field(None, description="Indicates facility affiliations. One or more of these")
    description: Optional[str] = Field(
        None, description="A brief paragraph describing the facility's services and/or history"
    )
    area: Optional[int] = Field(
        None, description="Total floor area of the facility in square meters"
    )
    numberDoctors: Optional[int] = Field(
        None, description="Total number of medical doctors working at the facility"
    )
    capacity: Optional[int] = Field(
        None, description="Overall inpatient bed capacity of the facility"
    )
    
    # -------------------------------------------------------------------
    # Appended from free_form.py
    # -------------------------------------------------------------------
    procedure: Optional[List[str]] = Field(
        None,
        description=(
            "Specific clinical services performed at the facility—medical/surgical interventions "
            "and diagnostic procedures and screenings (e.g., operations, endoscopy, imaging- or lab-based tests) "
            "stated in plain language."
        )
    )
    equipment: Optional[List[str]] = Field(
        None,
        description=(
            "Physical medical devices and infrastructure—imaging machines (MRI/CT/X-ray), surgical/OR technologies, "
            "monitors, laboratory analyzers, and critical utilities (e.g., piped oxygen/oxygen plants, backup power). "
            "Include specific models when available. Do NOT list bed counts here; only list specific bed devices/models."
        )
    )
    capability: Optional[List[str]] = Field(
        None,
        description=(
            "Medical capabilities defining what level and types of clinical care the facility can deliver—"
            "trauma/emergency care levels, specialized units (ICU/NICU/burn unit), clinical programs (stroke care, IVF), "
            "diagnostic capabilities (MRI, neurodiagnostics), accreditations, inpatient/outpatient, staffing levels, patient capacity. "
            "Excludes: addresses, contact info, business hours, pricing."
        )
    )
    
    # -------------------------------------------------------------------
    # Appended from medical_specialties.py
    # -------------------------------------------------------------------
    specialties: Optional[List[str]] = Field(
        None, description="The medical specialties associated with the organization"
    )


class NGO(BaseOrganization):
    """Pydantic model for NGO structured output extraction."""
    countries: Optional[List[str]] = Field(
        None, description="Countries where the NGO operates. (array of ISO alpha-2 codes)"
    )
    missionStatement: Optional[str] = Field(None, description="The NGO's formal mission statement")
    missionStatementLink: Optional[str] = Field(
        None, description="A url to the NGO's published mission statement"
    )
    organizationDescription: Optional[str] = Field(
        None,
        description="A neutral, factual description derived from the mission statement (removes explicitly religious or subjective language)",
    )


# -------------------------------------------------------------------
# Adapting organization_extraction.py for deep extraction
# Instead of abstract string lists, we embed the parsed classes
# -------------------------------------------------------------------
class DocumentExtraction(BaseModel):
    """
    Master schema for unified document extraction. 
    Combines the organization extraction with detailed deep extraction models.
    """
    ngos: Optional[List[NGO]] = Field(
        default_factory=list,
        description="Detailed profiles of NGOs explicitly mentioned in the text."
    )
    facilities: Optional[List[Facility]] = Field(
        default_factory=list,
        description="Detailed profiles of Healthcare facilities explicitly mentioned in the text."
    )
    other_organizations: Optional[List[BaseOrganization]] = Field(
        default_factory=list,
        description="Named entities that don't meet facility or NGO classifications, but have some basic details.",
    )
