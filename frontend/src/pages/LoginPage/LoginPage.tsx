import "./LoginPage.css";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { userRegister, userLogin } from "../../utility/api";
import { toast } from "react-toastify";
import "../../css/toasty.css";

const LoginPageDetails = [
  { label: "UserName", type: "text", placeholder: "Name" },
  { label: "Email", type: "email", placeholder: "Email" },
  { label: "Password", type: "password", placeholder: "Password" },
  { label: "contactNumber", type: "number", placeholder: "contact Number" },
  { label: "Role", type: "", placeholder: "" },
];

type PageState = "Login" | "SignUp";

export function LoginPage() {
  const [pageState, SetpageState] = useState<PageState>("Login");
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    contactnumber: "",
    role: "",
  });

  const fieldsToShow = LoginPageDetails.filter((detail) => {
    if (pageState === "Login") {
      return (
        detail.label !== "UserName" &&
        detail.label !== "Role" &&
        detail.label !== "contactNumber"
      );
    }
    return true;
  });

  const togglePage = () => {
    SetpageState((prev) => (prev === "Login" ? "SignUp" : "Login"));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const res =
        pageState === "SignUp"
          ? await userRegister(formData)
          : await userLogin(formData);

          console.log("res", res)

      if (res.success) {
        toast.success(res.message, {
          position: "top-right",
          autoClose: 3000,
          theme: "dark",
          className: "custom-toast-success",
        });

        if (res.data?.user.role === "collector") {
          localStorage.setItem("collecter_id", res.data?.user.id);
          navigate("/collector");
        } else if (res.data?.user.role === "admin") {
          navigate("/admin");
        } else {
          navigate("/reporter/wastage-classifier");
        }
      } else {
        navigate("/reporter/wastage-classifier");
        toast.error(res.message || "Something went wrong!", {
          position: "top-right",
          autoClose: 3000,
          theme: "dark",
          className: "custom-toast-error",
        });
      }
    } catch (error) {
      console.error("Error during authentication:", error);
      toast.error("An unexpected error occurred!", {
        position: "top-right",
        autoClose: 3000,
        theme: "dark",
        className: "custom-toast-error",
      });
    }
  };

  return (
    <div
      className="Login-page-bg"
      style={{ height: pageState === "SignUp" ? "580px" : "440px" }}
    >
      <form className="login-page-container" onSubmit={handleSubmit}>
        <div className="login-page-input">
          {fieldsToShow.map((detail, index) => (
            <>
              <label className="login-page-input" key={index}>
                {detail.label}
              </label>
              {detail.label === "Role" && pageState === "SignUp" ? (
                <div key={index} className="role-radio-group">
                  <label>
                    <input
                      type="radio"
                      name="role"
                      value="reporter"
                      required
                      checked={formData.role === "reporter"}
                      onChange={(e) =>
                        setFormData((prev) => ({
                          ...prev,
                          role: e.target.value,
                        }))
                      }
                    />{" "}
                    Reporter
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="role"
                      value="collector"
                      required
                      checked={formData.role == "collector"}
                      onChange={(e) =>
                        setFormData((prev) => ({
                          ...prev,
                          role: e.target.value,
                        }))
                      }
                    />
                    Collector
                  </label>
                </div>
              ) : (
                <input
                  type={detail.placeholder}
                  required
                  placeholder={detail.placeholder}
                  key={index}
                  value={
                    formData[
                      detail.label.toLocaleLowerCase() as keyof typeof formData
                    ]
                  }
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      [detail.label.toLocaleLowerCase()]: e.target.value,
                    }))
                  }
                />
              )}
            </>
          ))}
        </div>
        {pageState === "Login" ? (
          <>
            <div>
              Don't have an account?{" "}
              <span onClick={togglePage} className="toggle-link">
                SignUp
              </span>
            </div>
            <button type="submit" className="">
              {pageState}
            </button>
          </>
        ) : (
          <>
            <div>
              {" "}
              Already have an account?{" "}
              <span onClick={togglePage} className="toggle-link">
                Login
              </span>
            </div>
            <button type="submit" className="">
              {pageState}
            </button>
          </>
        )}
      </form>
    </div>
  );
}
