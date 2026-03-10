from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class FacilityType(str, Enum):
    hospital = "hospital"
    pharmacy = "pharmacy"
    doctor = "doctor"
    clinic = "clinic"
    dentist = "dentist"

class OperatorType(str, Enum):
    public = "public"
    private = "private"

class AffiliationType(str, Enum):
    faith_tradition = "faith-tradition"
    philanthropy_legacy = "philanthropy-legacy"
    community = "community"
    academic = "academic"
    government = "government"

class BaseOrganization(BaseModel):
    """Shared fields for all organizations."""
    # Contact Information
    name: str = Field(description="Official name of the organization. Use complete, unabbreviated form.")
    phone_numbers: List[str] = Field(default_factory=list, description="Phone numbers in E164 format (e.g. '+233392022664').")
    officialPhone: Optional[str] = Field(None, description="Official phone number in E164 format.")
    email: Optional[str] = Field(None, description="Primary email address.")
    
    # Web Presence
    websites: List[str] = Field(default_factory=list, description="Websites associated with the organization.")
    officialWebsite: Optional[str] = Field(None, description="Official website (domain name only).")
    facebookLink: Optional[str] = Field(None, description="URL to Facebook page.")
    twitterLink: Optional[str] = Field(None, description="URL to Twitter profile.")
    linkedinLink: Optional[str] = Field(None, description="URL to LinkedIn page.")
    instagramLink: Optional[str] = Field(None, description="Instagram account URL.")
    logo: Optional[str] = Field(None, description="URL linking to logo image.")
    
    # General
    yearEstablished: Optional[int] = Field(None, description="Year organization was established.")
    acceptsVolunteers: Optional[bool] = Field(None, description="Whether the organization accepts clinical volunteers.")
    
    # Address
    address_line1: Optional[str] = Field(None, description="Street address only (building number, street name).")
    address_line2: Optional[str] = Field(None, description="Additional street address information.")
    address_line3: Optional[str] = Field(None, description="Third line of street address if needed.")
    address_city: Optional[str] = Field(None, description="City or town name.")
    address_stateOrRegion: Optional[str] = Field(None, description="State, region, or province.")
    address_zipOrPostcode: Optional[str] = Field(None, description="ZIP or postal code.")
    address_country: Optional[str] = Field(None, description="Full country name.")
    address_countryCode: Optional[str] = Field(None, description="ISO alpha-2 country code. REQUIRED when country is known.")

class Facility(BaseOrganization):
    """Fields specific to healthcare facilities."""
    facilityTypeId: Optional[FacilityType] = Field(None, description="Type of facility.")
    operatorTypeId: Optional[OperatorType] = Field(None, description="Privately or publicly operated.")
    affiliationTypeIds: List[AffiliationType] = Field(default_factory=list, description="Facility affiliations.")
    description: Optional[str] = Field(None, description="Brief paragraph describing services/history.")
    area: Optional[int] = Field(None, description="Total floor area in square meters.")
    numberDoctors: Optional[int] = Field(None, description="Total number of medical doctors working at the facility.")
    capacity: Optional[int] = Field(None, description="Overall inpatient bed capacity.")
    
    # Facts - The prompt lists procedure, equipment, capability under 'Facility Facts' 
    # but associating them directly with the facility makes sense.
    procedure: List[str] = Field(default_factory=list, description="Specific clinical services performed.")
    equipment: List[str] = Field(default_factory=list, description="Physical medical devices and infrastructure.")
    capability: List[str] = Field(default_factory=list, description="Medical capabilities defining level of care.")
    
    # Medical Specialties
    specialties: List[str] = Field(default_factory=list, description="Medical specialties. Exact case-sensitive matches from hierarchy.")

class NGO(BaseOrganization):
    """Fields specific to NGOs."""
    countries: List[str] = Field(default_factory=list, description="Countries where NGO operates (ISO alpha-2 codes).")
    missionStatement: Optional[str] = Field(None, description="Formal mission statement.")
    missionStatementLink: Optional[str] = Field(None, description="URL to published mission statement.")
    organizationDescription: Optional[str] = Field(None, description="Neutral, factual description derived from mission statement.")

class DocumentExtraction(BaseModel):
    """Master schema for extracting data from a medical document."""
    ngos: List[NGO] = Field(default_factory=list, description="NGOs present in the text.")
    facilities: List[Facility] = Field(default_factory=list, description="Healthcare facilities present in the text.")
    other_organizations: List[BaseOrganization] = Field(default_factory=list, description="Named entities that don't meet facility or NGO classifications.")
