import pytest
from pydantic import ValidationError
from src.schema.models import Facility, NGO, DocumentExtraction, BaseOrganization

def test_facility_creation():
    fac = Facility(
        name="Korle-Bu Teaching Hospital",
        facilityTypeId="hospital",
        affiliationTypeIds=["academic", "government"],
        capacity=2000,
        address_city="Accra",
        address_country="Ghana",
        address_countryCode="GH",
        procedure=["Appendectomy"],
        equipment=["MRI Scanner"],
        specialties=["internalMedicine"]
    )
    assert fac.name == "Korle-Bu Teaching Hospital"
    assert fac.capacity == 2000
    assert fac.address_countryCode == "GH"
    assert len(fac.affiliationTypeIds) == 2
    assert fac.procedure == ["Appendectomy"]

def test_ngo_creation():
    ngo = NGO(
        name="Virtue Foundation",
        countries=["GH", "US"],
        missionStatement="To improve healthcare access."
    )
    assert ngo.name == "Virtue Foundation"
    assert "GH" in ngo.countries

def test_document_extraction():
    doc = DocumentExtraction(
        ngos=[
            NGO(name="Test NGO", countries=["GH"])
        ],
        facilities=[
            Facility(name="Test Clinic", facilityTypeId="clinic")
        ],
        other_organizations=[]
    )
    assert len(doc.ngos) == 1
    assert doc.ngos[0].name == "Test NGO"
    assert len(doc.facilities) == 1
    assert doc.facilities[0].facilityTypeId == "clinic"

def test_validation_errors():
    with pytest.raises(ValidationError):
        # Missing required name field
        BaseOrganization(phone_numbers=["+1234567890"])

