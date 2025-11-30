import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict
import json

class NotificationService:
    def __init__(self, smtp_server: str = None, smtp_port: int = None, 
                 email: str = None, password: str = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
    
    def send_email(self, to_emails: List[str], subject: str, body: str, 
                   html_body: str = None) -> bool:
        """
        Send an email to the specified recipients
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            # Add plain text body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML body if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Connect to server and send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  # Enable encryption
            server.login(self.email, self.password)
            
            text = msg.as_string()
            server.sendmail(self.email, to_emails, text)
            server.quit()
            
            print(f"Email sent successfully to {', '.join(to_emails)}")
            return True
            
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False
    
    def send_travel_alerts(self, user_email: str, alerts: List[Dict]) -> bool:
        """
        Send travel alerts (price drops, weather warnings, etc.) to user
        """
        if not alerts:
            return True  # Nothing to send
        
        subject = f"Travel Alerts - {len(alerts)} updates for your trips"
        
        body = "Here are the latest alerts for your travel plans:\n\n"
        for alert in alerts:
            body += f"- {alert.get('type', 'Alert')}: {alert.get('message', '')}\n"
            if 'date' in alert:
                body += f"  Date: {alert['date']}\n"
            body += "\n"
        
        body += "\nSafe travels!\nYour Travel Agent"
        
        return self.send_email([user_email], subject, body)
    
    def send_booking_confirmation(self, user_email: str, booking_details: Dict) -> bool:
        """
        Send booking confirmation to user
        """
        subject = f"Booking Confirmation - {booking_details.get('booking_id', 'N/A')}"
        
        body = f"""
        Dear Traveler,
        
        Your booking has been confirmed!
        
        Booking Details:
        - Booking ID: {booking_details.get('booking_id', 'N/A')}
        - Service: {booking_details.get('service_type', 'N/A')}
        - Provider: {booking_details.get('provider', 'N/A')}
        - Total Cost: {booking_details.get('total_cost', 'N/A')}
        - Dates: {booking_details.get('dates', 'N/A')}
        
        Please keep this information for your records.
        
        Safe travels!
        Your Travel Agent
        """
        
        return self.send_email([user_email], subject, body)

class TravelNotificationService(NotificationService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def send_price_drop_alert(self, user_email: str, service_type: str, 
                            original_price: float, new_price: float, 
                            service_details: Dict) -> bool:
        """
        Send alert when there's a price drop for a service the user is interested in
        """
        subject = f"Price Drop Alert! {service_type.title()} is now cheaper"
        
        body = f"""
        Great news! The {service_type} you were interested in has dropped in price.
        
        Previous Price: ${original_price:.2f}
        New Price: ${new_price:.2f}
        You can save: ${original_price - new_price:.2f}
        
        Service Details:
        - {service_details.get('name', service_details.get('airline', 'N/A'))}
        - {service_details.get('description', '')}
        
        Consider booking now before the price goes back up!
        
        Best regards,
        Your Travel Agent
        """
        
        return self.send_email([user_email], subject, body)
    
    def send_weather_alert(self, user_email: str, destination: str, 
                          weather_warning: str) -> bool:
        """
        Send weather alert for a destination
        """
        subject = f"Weather Alert for {destination}"
        
        body = f"""
        Important weather information for your trip to {destination}:
        
        {weather_warning}
        
        Please consider adjusting your travel plans accordingly.
        
        Stay safe,
        Your Travel Agent
        """
        
        return self.send_email([user_email], subject, body)