import email
import imaplib
import time
import pandas as pd
from similarity import Similarity
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib


emailID = 'emailtester108108@gmail.com'
password = 'yieqpjyarhqflcpx'


class ProcessEmail:
    def __init__(self, corpusName = None, test = False):
        emails = pd.read_csv(r"D:\OneDrive - Bansilal Ramnath Agarwal Charitable Trust, Vishwakarma Institute's\Previous Semesters\5th_Semester\EDI\myEmails.csv")
        sim_obj = Similarity(emails)
        self.login()
        self.corpusName = corpusName
        run = True
        last_email_id = 0
        self.models = None
        last_email = ""
        while run:
            replied = False
            rec_email_obj = ReciveEmail()
            msgSubject, msgFrom, msgBody = rec_email_obj.check_email()
            if ((self.user not in msgFrom) and
                    ("Auto generated Response:" not in msgSubject) and
                    (last_email != msgBody)):
                last_email = msgBody
                fout = open('lastEmail.txt', 'w')
                fout.write("Subject: "+msgSubject+"\n From: "+msgFrom+"\n Body: "+msgBody)
                fout.close()
                email_reply_body,similarity_value = sim_obj.rec_email_process(msgFrom,msgSubject,msgBody)
                if (similarity_value >= 0.30):
                    reply = ("This is an Auto generated Response" + "\n" +
                     "Subject: " +'Reply ' +msgSubject + '\n' +
                     "To: " + msgFrom + '\n' +
                     "From: " + "Auto reply: "+emailID + '\n\n' + email_reply_body)
                    send_mail(to=msgFrom,sub="This is an Auto generated Response", body=reply)
                    replied = True
                else:
                    print("No similar email is found hence no reply sent")
                if replied:
                    print ("The time is " + str(time.strftime("%I:%M:%S"))+" Email reply sent")
            else:
                print ("The time is " + str(time.strftime("%I:%M:%S")) +". No new mail. System will check again in 30 seconds.")
            if test:
                run = False
                break
            time.sleep(30)



    def login(self):
        self.user = emailID
        self.pswd = password
        self.mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        self.mail.login(self.user, self.pswd)


class ReciveEmail:
    def __init__(self,emailID = emailID,password=password):
        username = emailID
        password = password
        self.imap = imaplib.IMAP4_SSL("imap.gmail.com")
        self.imap.login(username, password)

        self.status, self.messages = self.imap.select("INBOX")
        self.N = 1
        self.messages = int(self.messages[0])

    def clean(self,text):
        return "".join(c if c.isalnum() else "_" for c in text)

    def check_email(self):
        for i in range(self.messages, self.messages-self.N, -1):
            res, msg = self.imap.fetch(str(i), "(RFC822)")
            for response in msg:
                if isinstance(response, tuple):
                    msg = email.message_from_bytes(response[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding)
                    From, encoding = decode_header(msg.get("From"))[0]
                    if isinstance(From, bytes):
                        From = From.decode(encoding)
                    print("Subject:", subject)
                    print("From:", From)
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            try:
                                body = part.get_payload(decode=True).decode()
                            except:
                                pass
                            if content_type == "text/plain" and "attachment" not in content_disposition:
                                print("Body of the Email: ",body)
                            elif "attachment" in content_disposition:
                                filename = part.get_filename()
                    else:
                        content_type = msg.get_content_type()
                        body = msg.get_payload(decode=True).decode()
                        if content_type == "text/plain":
                            print(body)
                    if content_type == "text/html":
                        folder_name = self.clean(subject)
                    print("="*100)
        self.imap.close()
        self.imap.logout()
        return subject, From, body
    

def message(subject="Send_email test Subject", 
            text="This is the Body for send_email program to test the send_email functionality for sedning emails through python script", img=None,
            attachment=None):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg.attach(MIMEText(text))
    return msg


def send_mail(to, sub, body, emailID = emailID, password=password):
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login(emailID, password)
    msg = message(subject=sub, text=body)
    to = to
    smtp.sendmail(from_addr=emailID, to_addrs=to, msg=msg.as_string())
    smtp.quit()


if __name__ == '__main__':
    corpus ="myemail"
    proc = ProcessEmail(corpus)