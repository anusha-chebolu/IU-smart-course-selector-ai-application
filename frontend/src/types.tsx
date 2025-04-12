export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Chat {
  id: string;
  messages: Message[];
  name: string;
}

export interface User {
  id: string;
  username: string;
  firstName: string;
  lastName: string;
  email: string;
}

// types.ts
export interface Course {
  course_id: string;
  course_name: string;
  department: string;
  min_credits: number;
  max_credits: number;
  prerequisites: string[];
  offered_semester: string;
  course_title: string;
  course_description: string;
  course_details: {
    classes: Array<{
      car: string;
      moi: string;
      inst: string;
      strm: string;
      campus: string;
      closed: boolean;
      acadOrg: string;
      created: number;
      primary: boolean;
      classNbr: number;
      courseId: string;
      location: string;
      maxUnits: number;
      minUnits: number;
      classType: string;
      openSeats: number;
      totalSeats: number;
      description: string | null;
      sessionDescr: string;
      modeOfInstruction: string;
      locationDescription: string;
      primaryInstructors: Array<{
        role: string;
        fullName: string;
        lastName: string;
        firstName: string;
        middleName: string | null;
      }>;
      meetings: Array<{
        // Meeting details
        room: string;
        meetingDays: string;
        component: string;
        buildingName: string;
        classSection: string;
      }>;
    }>;
    attributes: string[];
    institution: string;
    courseTopicId: number;
    effectiveDate: number;
    academicCareer: string;
    courseOfferNumber: number;
  };
}

export interface UserChat {
  id: string;
  userId: string;
  title?: string;
  lastInteracted?: string;
}

export interface ChatMessage {
  id?: string;
  chatId: string;
  userId: string;
  question: string;
  response: string;
  created_at?: string;
}

export interface User {
  id: string;
  username: string;
  password?: string;
  email: string;
  firstName: string;
  lastName: string;
  createdAt?: string;
  chats?: {
    id: string;
    messages: ChatMessage[];
    title?: string;
    lastInteracted: string;
  }[];
}
