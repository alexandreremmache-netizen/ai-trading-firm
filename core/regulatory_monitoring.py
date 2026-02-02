"""
Regulatory Change Monitoring and Compliance Report Templates.

This module provides automated monitoring of regulatory changes and
standardized templates for compliance reporting.

Addresses:
- #C42 - Regulatory change monitoring not automated
- #C43 - Compliance report templates outdated
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import json


logger = logging.getLogger(__name__)


class RegulatorType(Enum):
    """Types of financial regulators."""
    AMF = "amf"  # Autorite des Marches Financiers (France)
    ESMA = "esma"  # European Securities and Markets Authority
    FCA = "fca"  # Financial Conduct Authority (UK)
    SEC = "sec"  # Securities and Exchange Commission (US)
    CFTC = "cftc"  # Commodity Futures Trading Commission (US)
    BIS = "bis"  # Bank for International Settlements
    ECB = "ecb"  # European Central Bank


class RegulatoryDomain(Enum):
    """Regulatory domains/categories."""
    MARKET_ABUSE = "market_abuse"
    TRANSACTION_REPORTING = "transaction_reporting"
    BEST_EXECUTION = "best_execution"
    POSITION_LIMITS = "position_limits"
    CAPITAL_REQUIREMENTS = "capital_requirements"
    ALGORITHMIC_TRADING = "algorithmic_trading"
    DATA_PROTECTION = "data_protection"
    AML_KYC = "aml_kyc"
    RECORD_KEEPING = "record_keeping"
    CLIENT_PROTECTION = "client_protection"


class ChangeImpact(Enum):
    """Impact level of regulatory changes."""
    CRITICAL = "critical"  # Requires immediate action
    HIGH = "high"  # Requires action within 30 days
    MEDIUM = "medium"  # Requires action within 90 days
    LOW = "low"  # Informational, no immediate action
    INFORMATIONAL = "informational"  # No action required


class ChangeStatus(Enum):
    """Status of regulatory change tracking."""
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    IMPACT_ASSESSED = "impact_assessed"
    IMPLEMENTATION_PLANNED = "implementation_planned"
    IMPLEMENTING = "implementing"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


@dataclass
class RegulatoryChange:
    """Represents a regulatory change or update."""
    change_id: str
    title: str
    regulator: RegulatorType
    domain: RegulatoryDomain
    description: str
    effective_date: datetime
    publication_date: datetime
    impact: ChangeImpact
    status: ChangeStatus = ChangeStatus.PENDING_REVIEW
    source_url: str = ""
    affected_modules: List[str] = field(default_factory=list)
    implementation_notes: str = ""
    assigned_to: str = ""
    review_deadline: Optional[datetime] = None
    implementation_deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_id": self.change_id,
            "title": self.title,
            "regulator": self.regulator.value,
            "domain": self.domain.value,
            "description": self.description,
            "effective_date": self.effective_date.isoformat(),
            "publication_date": self.publication_date.isoformat(),
            "impact": self.impact.value,
            "status": self.status.value,
            "source_url": self.source_url,
            "affected_modules": self.affected_modules,
            "implementation_notes": self.implementation_notes,
            "assigned_to": self.assigned_to,
            "review_deadline": self.review_deadline.isoformat() if self.review_deadline else None,
            "implementation_deadline": self.implementation_deadline.isoformat() if self.implementation_deadline else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def days_until_effective(self) -> int:
        """Calculate days until effective date."""
        delta = self.effective_date - datetime.now(timezone.utc)
        return max(0, delta.days)

    def is_overdue(self) -> bool:
        """Check if implementation is overdue."""
        if self.status == ChangeStatus.VERIFIED:
            return False
        if self.implementation_deadline:
            return datetime.now(timezone.utc) > self.implementation_deadline
        return datetime.now(timezone.utc) > self.effective_date


@dataclass
class RegulatoryFeed:
    """Configuration for a regulatory feed source."""
    feed_id: str
    regulator: RegulatorType
    feed_url: str
    feed_type: str  # rss, api, scrape
    check_interval_hours: int = 24
    last_checked: Optional[datetime] = None
    enabled: bool = True
    auth_required: bool = False
    auth_config: Dict[str, str] = field(default_factory=dict)


class RegulatoryChangeMonitor:
    """
    Monitors regulatory changes from various sources.

    Provides automated tracking and alerting for regulatory updates
    that may affect trading operations.
    """

    def __init__(self):
        self.feeds: Dict[str, RegulatoryFeed] = {}
        self.changes: Dict[str, RegulatoryChange] = {}
        self.subscribers: Dict[RegulatoryDomain, List[Callable]] = {}
        self.seen_hashes: Set[str] = set()
        self._setup_default_feeds()

    def _setup_default_feeds(self):
        """Configure default regulatory feed sources."""
        default_feeds = [
            RegulatoryFeed(
                feed_id="esma_news",
                regulator=RegulatorType.ESMA,
                feed_url="https://www.esma.europa.eu/press-news/esma-news/feed",
                feed_type="rss",
                check_interval_hours=12
            ),
            RegulatoryFeed(
                feed_id="amf_news",
                regulator=RegulatorType.AMF,
                feed_url="https://www.amf-france.org/en/news-publications/news-releases/feed",
                feed_type="rss",
                check_interval_hours=12
            ),
            RegulatoryFeed(
                feed_id="fca_news",
                regulator=RegulatorType.FCA,
                feed_url="https://www.fca.org.uk/news/rss.xml",
                feed_type="rss",
                check_interval_hours=12
            )
        ]
        for feed in default_feeds:
            self.feeds[feed.feed_id] = feed

    def add_feed(self, feed: RegulatoryFeed):
        """Add a regulatory feed source."""
        self.feeds[feed.feed_id] = feed
        logger.info(f"Added regulatory feed: {feed.feed_id}")

    def subscribe(self, domain: RegulatoryDomain, callback: Callable[[RegulatoryChange], None]):
        """Subscribe to changes in a regulatory domain."""
        if domain not in self.subscribers:
            self.subscribers[domain] = []
        self.subscribers[domain].append(callback)

    def _generate_change_id(self, title: str, regulator: str, date: datetime) -> str:
        """Generate unique change ID."""
        content = f"{title}:{regulator}:{date.isoformat()}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"REG-{hash_val.upper()}"

    def _classify_domain(self, title: str, description: str) -> RegulatoryDomain:
        """Classify regulatory change into a domain based on content."""
        text = f"{title} {description}".lower()

        domain_keywords = {
            RegulatoryDomain.MARKET_ABUSE: ["mar", "market abuse", "insider", "manipulation"],
            RegulatoryDomain.TRANSACTION_REPORTING: ["emir", "reporting", "transaction report", "trade repository"],
            RegulatoryDomain.BEST_EXECUTION: ["best execution", "rts 27", "rts 28", "execution quality"],
            RegulatoryDomain.POSITION_LIMITS: ["position limit", "commodity derivative"],
            RegulatoryDomain.CAPITAL_REQUIREMENTS: ["capital", "crr", "crd", "prudential"],
            RegulatoryDomain.ALGORITHMIC_TRADING: ["algorithmic", "algo trading", "rts 6", "hft"],
            RegulatoryDomain.DATA_PROTECTION: ["gdpr", "data protection", "privacy"],
            RegulatoryDomain.AML_KYC: ["aml", "kyc", "money laundering", "terrorist financing"],
            RegulatoryDomain.RECORD_KEEPING: ["record keeping", "retention", "rts 25"],
            RegulatoryDomain.CLIENT_PROTECTION: ["client", "investor protection", "suitability"]
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                return domain

        return RegulatoryDomain.TRANSACTION_REPORTING  # Default

    def _assess_impact(self, change: RegulatoryChange) -> ChangeImpact:
        """Assess the impact level of a regulatory change."""
        days_until = change.days_until_effective()

        # Critical if already effective or within 7 days
        if days_until <= 7:
            return ChangeImpact.CRITICAL

        # High if within 30 days
        if days_until <= 30:
            return ChangeImpact.HIGH

        # Medium if within 90 days
        if days_until <= 90:
            return ChangeImpact.MEDIUM

        # Otherwise low
        return ChangeImpact.LOW

    def register_change(self, change: RegulatoryChange) -> bool:
        """
        Register a new regulatory change.

        Returns True if this is a new change, False if duplicate.
        """
        # Check for duplicates
        content_hash = hashlib.md5(
            f"{change.title}:{change.regulator.value}:{change.publication_date.isoformat()}".encode()
        ).hexdigest()

        if content_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(content_hash)

        # Auto-assess impact if not set
        if change.impact == ChangeImpact.INFORMATIONAL:
            change.impact = self._assess_impact(change)

        # Set review deadline based on impact
        if not change.review_deadline:
            review_days = {
                ChangeImpact.CRITICAL: 1,
                ChangeImpact.HIGH: 7,
                ChangeImpact.MEDIUM: 14,
                ChangeImpact.LOW: 30,
                ChangeImpact.INFORMATIONAL: 60
            }
            change.review_deadline = datetime.now(timezone.utc) + timedelta(
                days=review_days.get(change.impact, 30)
            )

        self.changes[change.change_id] = change

        # Notify subscribers
        if change.domain in self.subscribers:
            for callback in self.subscribers[change.domain]:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")

        logger.info(f"Registered regulatory change: {change.change_id} - {change.title}")
        return True

    def update_status(self, change_id: str, status: ChangeStatus, notes: str = ""):
        """Update the status of a regulatory change."""
        if change_id not in self.changes:
            raise ValueError(f"Change not found: {change_id}")

        change = self.changes[change_id]
        old_status = change.status
        change.status = status
        change.updated_at = datetime.now(timezone.utc)

        if notes:
            change.implementation_notes += f"\n[{datetime.now(timezone.utc).isoformat()}] {notes}"

        logger.info(f"Updated change {change_id} status: {old_status.value} -> {status.value}")

    def get_pending_changes(self,
                           impact_filter: Optional[ChangeImpact] = None,
                           domain_filter: Optional[RegulatoryDomain] = None) -> List[RegulatoryChange]:
        """Get list of pending regulatory changes."""
        changes = []

        for change in self.changes.values():
            if change.status in [ChangeStatus.VERIFIED, ChangeStatus.IMPLEMENTED]:
                continue

            if impact_filter and change.impact != impact_filter:
                continue

            if domain_filter and change.domain != domain_filter:
                continue

            changes.append(change)

        # Sort by effective date
        changes.sort(key=lambda c: c.effective_date)
        return changes

    def get_overdue_changes(self) -> List[RegulatoryChange]:
        """Get list of overdue regulatory changes."""
        return [c for c in self.changes.values() if c.is_overdue()]

    def get_change_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked regulatory changes."""
        by_status = {}
        by_impact = {}
        by_domain = {}

        for change in self.changes.values():
            by_status[change.status.value] = by_status.get(change.status.value, 0) + 1
            by_impact[change.impact.value] = by_impact.get(change.impact.value, 0) + 1
            by_domain[change.domain.value] = by_domain.get(change.domain.value, 0) + 1

        overdue = self.get_overdue_changes()

        return {
            "total_changes": len(self.changes),
            "by_status": by_status,
            "by_impact": by_impact,
            "by_domain": by_domain,
            "overdue_count": len(overdue),
            "overdue_changes": [c.change_id for c in overdue],
            "feeds_configured": len(self.feeds),
            "active_feeds": sum(1 for f in self.feeds.values() if f.enabled)
        }


# ============================================================================
# Compliance Report Templates
# ============================================================================

class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    XML = "xml"


@dataclass
class ReportSection:
    """Section within a compliance report."""
    section_id: str
    title: str
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    subsections: List['ReportSection'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "data": self.data,
            "subsections": [s.to_dict() for s in self.subsections]
        }


@dataclass
class ComplianceReportTemplate:
    """Template for compliance reports."""
    template_id: str
    name: str
    description: str
    regulation: str  # e.g., "MiFID II", "EMIR", "MAR"
    report_type: str  # e.g., "periodic", "ad-hoc", "regulatory"
    frequency: str  # e.g., "daily", "weekly", "monthly", "quarterly", "annual"
    sections: List[str] = field(default_factory=list)
    required_data: List[str] = field(default_factory=list)
    output_formats: List[ReportFormat] = field(default_factory=list)
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "regulation": self.regulation,
            "report_type": self.report_type,
            "frequency": self.frequency,
            "sections": self.sections,
            "required_data": self.required_data,
            "output_formats": [f.value for f in self.output_formats],
            "version": self.version,
            "last_updated": self.last_updated.isoformat()
        }


class ComplianceReportGenerator:
    """
    Generates compliance reports using standardized templates.

    Provides templates for various EU regulatory requirements including
    MiFID II, EMIR, MAR, and internal compliance reporting.
    """

    def __init__(self):
        self.templates: Dict[str, ComplianceReportTemplate] = {}
        self._setup_default_templates()

    def _setup_default_templates(self):
        """Set up default compliance report templates."""
        templates = [
            # MiFID II RTS 27 Best Execution Report
            ComplianceReportTemplate(
                template_id="rts27_best_execution",
                name="RTS 27 Best Execution Report",
                description="Quarterly report on execution quality per MiFID II RTS 27",
                regulation="MiFID II",
                report_type="regulatory",
                frequency="quarterly",
                sections=[
                    "executive_summary",
                    "venue_statistics",
                    "price_quality",
                    "speed_execution",
                    "likelihood_execution",
                    "settlement_quality",
                    "transaction_costs"
                ],
                required_data=[
                    "executions_by_venue",
                    "execution_times",
                    "price_improvements",
                    "failed_settlements",
                    "transaction_costs"
                ],
                output_formats=[ReportFormat.PDF, ReportFormat.XML, ReportFormat.EXCEL]
            ),

            # MiFID II RTS 28 Venue Analysis Report
            ComplianceReportTemplate(
                template_id="rts28_venue_analysis",
                name="RTS 28 Top 5 Venues Report",
                description="Annual report on top execution venues per MiFID II RTS 28",
                regulation="MiFID II",
                report_type="regulatory",
                frequency="annual",
                sections=[
                    "methodology",
                    "top_venues_by_class",
                    "order_routing_decisions",
                    "conflicts_of_interest",
                    "passive_aggressive_analysis"
                ],
                required_data=[
                    "venue_volumes",
                    "order_routing_policies",
                    "inducements",
                    "fill_types"
                ],
                output_formats=[ReportFormat.PDF, ReportFormat.HTML]
            ),

            # EMIR Trade Reporting
            ComplianceReportTemplate(
                template_id="emir_trade_report",
                name="EMIR Trade Report",
                description="Daily transaction report for EMIR compliance",
                regulation="EMIR",
                report_type="regulatory",
                frequency="daily",
                sections=[
                    "reporting_summary",
                    "counterparty_details",
                    "trade_details",
                    "valuation_data",
                    "collateral_data"
                ],
                required_data=[
                    "trades",
                    "counterparties",
                    "valuations",
                    "collateral_movements"
                ],
                output_formats=[ReportFormat.XML]
            ),

            # MAR Suspicious Transaction Report
            ComplianceReportTemplate(
                template_id="mar_stor",
                name="Suspicious Transaction and Order Report (STOR)",
                description="Report for suspicious market activity per MAR Article 16",
                regulation="MAR",
                report_type="ad-hoc",
                frequency="as_needed",
                sections=[
                    "incident_summary",
                    "transaction_details",
                    "suspicion_indicators",
                    "supporting_evidence",
                    "risk_assessment"
                ],
                required_data=[
                    "suspicious_orders",
                    "market_data",
                    "detection_alerts",
                    "historical_patterns"
                ],
                output_formats=[ReportFormat.PDF, ReportFormat.XML]
            ),

            # Daily Compliance Summary
            ComplianceReportTemplate(
                template_id="daily_compliance_summary",
                name="Daily Compliance Summary",
                description="Internal daily compliance status report",
                regulation="Internal",
                report_type="periodic",
                frequency="daily",
                sections=[
                    "trading_activity",
                    "limit_utilization",
                    "breach_summary",
                    "pending_actions",
                    "upcoming_deadlines"
                ],
                required_data=[
                    "trades",
                    "positions",
                    "risk_limits",
                    "alerts"
                ],
                output_formats=[ReportFormat.HTML, ReportFormat.JSON]
            ),

            # Monthly Regulatory Report
            ComplianceReportTemplate(
                template_id="monthly_regulatory_summary",
                name="Monthly Regulatory Summary",
                description="Monthly summary of regulatory compliance status",
                regulation="Multi-regulation",
                report_type="periodic",
                frequency="monthly",
                sections=[
                    "executive_summary",
                    "mifid_compliance",
                    "emir_compliance",
                    "mar_compliance",
                    "incident_summary",
                    "action_items"
                ],
                required_data=[
                    "reporting_status",
                    "breaches",
                    "incidents",
                    "remediation_status"
                ],
                output_formats=[ReportFormat.PDF, ReportFormat.HTML]
            ),

            # Risk Report
            ComplianceReportTemplate(
                template_id="daily_risk_report",
                name="Daily Risk Report",
                description="Daily risk metrics and limit utilization report",
                regulation="Internal",
                report_type="periodic",
                frequency="daily",
                sections=[
                    "portfolio_summary",
                    "var_metrics",
                    "limit_utilization",
                    "stress_test_results",
                    "concentration_analysis"
                ],
                required_data=[
                    "positions",
                    "var_calculations",
                    "risk_limits",
                    "stress_scenarios"
                ],
                output_formats=[ReportFormat.HTML, ReportFormat.PDF, ReportFormat.JSON]
            ),

            # Algorithmic Trading Report (RTS 6)
            ComplianceReportTemplate(
                template_id="rts6_algo_trading",
                name="RTS 6 Algorithmic Trading Report",
                description="Report on algorithmic trading systems per MiFID II RTS 6",
                regulation="MiFID II",
                report_type="regulatory",
                frequency="annual",
                sections=[
                    "system_overview",
                    "algorithm_inventory",
                    "kill_switch_tests",
                    "pre_trade_controls",
                    "incident_log"
                ],
                required_data=[
                    "algorithms",
                    "system_tests",
                    "control_parameters",
                    "incidents"
                ],
                output_formats=[ReportFormat.PDF]
            )
        ]

        for template in templates:
            self.templates[template.template_id] = template

    def get_template(self, template_id: str) -> Optional[ComplianceReportTemplate]:
        """Get a report template by ID."""
        return self.templates.get(template_id)

    def list_templates(self,
                       regulation_filter: Optional[str] = None,
                       frequency_filter: Optional[str] = None) -> List[ComplianceReportTemplate]:
        """List available report templates."""
        templates = list(self.templates.values())

        if regulation_filter:
            templates = [t for t in templates if t.regulation == regulation_filter]

        if frequency_filter:
            templates = [t for t in templates if t.frequency == frequency_filter]

        return templates

    def generate_report(self,
                       template_id: str,
                       data: Dict[str, Any],
                       output_format: ReportFormat = ReportFormat.JSON,
                       report_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a compliance report from a template.

        Args:
            template_id: ID of the template to use
            data: Data to populate the report
            output_format: Desired output format
            report_date: Date for the report (defaults to today)

        Returns:
            Generated report content
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        if output_format not in template.output_formats:
            raise ValueError(f"Format {output_format.value} not supported for template {template_id}")

        report_date = report_date or datetime.now()

        # Validate required data
        missing_data = [d for d in template.required_data if d not in data]
        if missing_data:
            logger.warning(f"Missing required data for report: {missing_data}")

        # Build report structure
        report = {
            "report_id": f"{template_id}_{report_date.strftime('%Y%m%d_%H%M%S')}",
            "template_id": template_id,
            "template_name": template.name,
            "template_version": template.version,
            "regulation": template.regulation,
            "report_type": template.report_type,
            "report_date": report_date.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "sections": {}
        }

        # Populate sections
        for section in template.sections:
            section_data = self._build_section(section, data)
            report["sections"][section] = section_data

        # Add metadata
        report["metadata"] = {
            "data_completeness": len([d for d in template.required_data if d in data]) / len(template.required_data) if template.required_data else 1.0,
            "missing_data": missing_data,
            "output_format": output_format.value
        }

        logger.info(f"Generated report: {report['report_id']}")
        return report

    def _build_section(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a report section with available data."""
        # Map section names to data extraction logic
        section_builders = {
            "executive_summary": self._build_executive_summary,
            "trading_activity": self._build_trading_activity,
            "limit_utilization": self._build_limit_utilization,
            "breach_summary": self._build_breach_summary,
            "var_metrics": self._build_var_metrics,
            "venue_statistics": self._build_venue_statistics
        }

        builder = section_builders.get(section_name, self._build_generic_section)
        return builder(section_name, data)

    def _build_executive_summary(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        return {
            "title": "Executive Summary",
            "content": "Summary of key compliance metrics and status.",
            "data": {
                "total_trades": data.get("trades", {}).get("count", 0),
                "total_volume": data.get("trades", {}).get("volume", 0),
                "breaches": data.get("breaches", {}).get("count", 0),
                "alerts": data.get("alerts", {}).get("count", 0)
            }
        }

    def _build_trading_activity(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build trading activity section."""
        trades = data.get("trades", {})
        return {
            "title": "Trading Activity",
            "content": "Summary of trading activity during the reporting period.",
            "data": {
                "total_orders": trades.get("order_count", 0),
                "executed_orders": trades.get("executed_count", 0),
                "cancelled_orders": trades.get("cancelled_count", 0),
                "total_volume": trades.get("volume", 0),
                "by_instrument_type": trades.get("by_type", {}),
                "by_venue": trades.get("by_venue", {})
            }
        }

    def _build_limit_utilization(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build limit utilization section."""
        limits = data.get("risk_limits", {})
        return {
            "title": "Limit Utilization",
            "content": "Current utilization of risk and position limits.",
            "data": {
                "var_utilization": limits.get("var_utilization", 0),
                "position_utilization": limits.get("position_utilization", 0),
                "leverage_utilization": limits.get("leverage_utilization", 0),
                "concentration_utilization": limits.get("concentration_utilization", 0)
            }
        }

    def _build_breach_summary(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build breach summary section."""
        breaches = data.get("breaches", {})
        return {
            "title": "Breach Summary",
            "content": "Summary of limit breaches and violations.",
            "data": {
                "total_breaches": breaches.get("count", 0),
                "by_type": breaches.get("by_type", {}),
                "resolved": breaches.get("resolved", 0),
                "pending": breaches.get("pending", 0)
            }
        }

    def _build_var_metrics(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build VaR metrics section."""
        var_data = data.get("var_calculations", {})
        return {
            "title": "Value at Risk Metrics",
            "content": "Daily VaR and risk metrics.",
            "data": {
                "var_95": var_data.get("var_95", 0),
                "var_99": var_data.get("var_99", 0),
                "cvar_95": var_data.get("cvar_95", 0),
                "cvar_99": var_data.get("cvar_99", 0)
            }
        }

    def _build_venue_statistics(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build venue statistics section."""
        venues = data.get("executions_by_venue", {})
        return {
            "title": "Venue Statistics",
            "content": "Execution statistics by trading venue.",
            "data": {
                "venues": venues,
                "total_venues": len(venues)
            }
        }

    def _build_generic_section(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a generic section."""
        return {
            "title": section_name.replace("_", " ").title(),
            "content": f"Data for {section_name}",
            "data": data.get(section_name, {})
        }

    def export_template_catalog(self) -> str:
        """Export catalog of all templates as JSON."""
        catalog = {
            "generated_at": datetime.now().isoformat(),
            "template_count": len(self.templates),
            "templates": [t.to_dict() for t in self.templates.values()]
        }
        return json.dumps(catalog, indent=2)


# ============================================================================
# Combined Regulatory Monitoring System
# ============================================================================

class RegulatoryComplianceSystem:
    """
    Unified system for regulatory change monitoring and compliance reporting.

    Combines change monitoring with report generation for comprehensive
    regulatory compliance management.
    """

    def __init__(self):
        self.change_monitor = RegulatoryChangeMonitor()
        self.report_generator = ComplianceReportGenerator()

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        change_summary = self.change_monitor.get_change_summary()

        return {
            "timestamp": datetime.now().isoformat(),
            "regulatory_changes": change_summary,
            "available_templates": len(self.report_generator.templates),
            "pending_changes": len(self.change_monitor.get_pending_changes()),
            "overdue_changes": len(self.change_monitor.get_overdue_changes())
        }

    def generate_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for compliance dashboard."""
        pending = self.change_monitor.get_pending_changes()
        overdue = self.change_monitor.get_overdue_changes()

        return {
            "summary": {
                "total_tracked_changes": len(self.change_monitor.changes),
                "pending_changes": len(pending),
                "overdue_changes": len(overdue),
                "critical_changes": len([c for c in pending if c.impact == ChangeImpact.CRITICAL])
            },
            "pending_changes": [c.to_dict() for c in pending[:10]],
            "overdue_changes": [c.to_dict() for c in overdue],
            "upcoming_deadlines": self._get_upcoming_deadlines(),
            "templates_available": len(self.report_generator.templates)
        }

    def _get_upcoming_deadlines(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get regulatory deadlines in the next N days."""
        cutoff = datetime.now() + timedelta(days=days)
        deadlines = []

        for change in self.change_monitor.changes.values():
            if change.effective_date <= cutoff and change.status != ChangeStatus.VERIFIED:
                deadlines.append({
                    "change_id": change.change_id,
                    "title": change.title,
                    "effective_date": change.effective_date.isoformat(),
                    "days_remaining": change.days_until_effective(),
                    "impact": change.impact.value
                })

        deadlines.sort(key=lambda d: d["days_remaining"])
        return deadlines
